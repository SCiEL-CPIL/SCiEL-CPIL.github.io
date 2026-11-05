import itertools
import time
import random
import math
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Uses GRASP (Greedy Randomized Adaptive Search Procedure).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def check_time(budget=2.0):
        return time.time() - start_time > deadline - budget

    # --- Precomputation ---
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build incidence structure
    # block_tsubs_indices[b] = list of t-subset indices covered by block b
    # tsub_blocks_indices[i] = list of block indices covering t-subset i
    block_tsubs_indices = [[] for _ in range(num_blocks)]
    tsub_blocks_indices = [[] for _ in range(num_tsubs)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            t_idx = tsub_index[ts]
            block_tsubs_indices[b_idx].append(t_idx)
            tsub_blocks_indices[t_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access if needed, but lists are fine for sparse logic
    # Degrees
    degs = np.array([len(tsub_blocks_indices[i]) for i in range(num_tsubs)], dtype=np.float64)
    degs[degs == 0] = 1.0
    base_weights = 1.0 / degs

    # --- GRASP Components ---
    
    def grasp_construct(alpha=0.5, weights=None):
        """
        Construct a feasible solution using randomized greedy.
        At each step, pick a block from the RCL (top candidates by score) randomly.
        """
        if weights is None:
            weights = base_weights
            
        uncovered = np.ones(num_tsubs, dtype=np.float64)
        selected = []
        uncovered_count = num_tsubs
        
        # Initial scores
        # score[b] = sum(weights[t] for t in block_tsubs_indices[b] if uncovered[t])
        # We can compute this efficiently.
        scores = np.zeros(num_blocks, dtype=np.float64)
        for b in range(num_blocks):
            # Only sum weights of currently uncovered tsubs in this block
            # Since initially all are uncovered, score[b] = sum(weights[t] for t in block_tsubs_indices[b])
            # But weights are static per run? No, weights are static for the run, 
            # but 'uncovered' changes.
            # Actually, standard GRASP usually uses static weights or updates them.
            # Let's use static weights for simplicity in RCL formation, 
            # but the "gain" should be the number of NEWLY covered tsubs.
            # A common heuristic is: score = sum(weights[t] * uncovered[t])
            
            # Let's precompute the full score vector once, and update it.
            pass
        
        # Efficient score initialization
        # score = A.T @ (weights * uncovered)
        # Since we don't have A as a matrix object handy for fast mat-vec without building it,
        # and building it takes time/memory, let's do it manually or build a sparse matrix.
        
        # Let's build the sparse matrix A once. It's C(19,5) x C(19,8) = 11628 x 75582.
        # nnz = C(19,8)*C(8,5) = 75582 * 56 = 4,232,592.
        # This is small enough.
        
        rows = []
        cols = []
        for b in range(num_blocks):
            for t_idx in block_tsubs_indices[b]:
                rows.append(t_idx)
                cols.append(b)
        
        A = lil_matrix((num_tsubs, num_blocks), dtype=np.float32)
        for r, c in zip(rows, cols):
            A[r, c] = 1.0
        A_csr = A.tocsr()
        A_csc = A.tocsc() # For column access? No, we need row sums mostly.
        
        # Actually, A_csr.T @ vec is fast.
        w_u = weights * uncovered
        scores = A_csr.T @ w_u
        
        selected_set = set()
        
        while uncovered_count > 0:
            if check_time(2.0):
                break
                
            # 1. Determine RCL
            # Find max score
            max_score = np.max(scores)
            if max_score <= 0:
                # Fallback: pick any block covering an uncovered tsub
                unc_idx = np.where(uncovered > 0)[0]
                if len(unc_idx) == 0: break
                target_t = unc_idx[0]
                candidates = tsub_blocks_indices[target_t]
                if not candidates: break
                b_idx = candidates[0]
            else:
                # RCL: blocks with score >= (1-alpha)*max + alpha*min? 
                # Or top K%? 
                # Standard: score >= (1-alpha)*max_score + alpha*min_score (among candidates)
                # But min_score is 0 for most. 
                # Better: score >= (1-alpha) * max_score.
                threshold = (1 - alpha) * max_score
                rcl_indices = np.where(scores >= threshold)[0]
                
                # Filter out already selected
                rcl_candidates = [i for i in rcl_indices if i not in selected_set]
                
                if not rcl_candidates:
                    # Fallback to max score unselected
                    valid_scores = scores.copy()
                    for b in selected_set:
                        valid_scores[b] = -1
                    b_idx = int(np.argmax(valid_scores))
                else:
                    # Random pick from RCL
                    b_idx = random.choice(rcl_candidates)
            
            selected_set.add(b_idx)
            selected.append(b_idx)
            
            # Update uncovered and scores
            tsubs_in_b = block_tsubs_indices[b_idx]
            for t_idx in tsubs_in_b:
                if uncovered[t_idx] > 0:
                    uncovered[t_idx] = 0
                    uncovered_count -= 1
            
            # Update scores: subtract contribution of newly covered tsubs
            # delta_scores = A[:, tsubs_in_b].T @ (weights * newly_covered_indicator)
            # Since we set uncovered to 0, the change in w_u is -weights[t] for these t's.
            # So scores -= A[:, tsubs_in_b].T @ weights[tsubs_in_b]
            
            # A[:, tsubs_in_b] is a sparse matrix slice. 
            # Using A_csr[tsubs_in_b, :] gets rows.
            # We need A[:, tsubs_in_b].T which is A[tsubs_in_b, :].T
            # Let's just recompute scores. It's O(nnz). 
            # 4M ops * ~300 blocks = 1.2B ops. Might be tight in Python loop?
            # NumPy sparse mat-vec is C optimized.
            
            scores = A_csr.T @ (weights * uncovered)
            
            # Mask selected blocks
            for b in selected_set:
                scores[b] = -1e9
                
        return selected

    def local_search(block_indices):
        """
        Refine solution by removing redundant blocks and swapping.
        """
        block_indices = list(block_indices)
        if not block_indices:
            return block_indices
            
        # 1. Redundant Removal
        # Calculate cover counts
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in block_indices:
            cover_count[block_tsubs_indices[b]] += 1
            
        changed = True
        while changed:
            if check_time(3.0):
                break
            changed = False
            new_blocks = []
            for b in block_indices:
                tsubs = block_tsubs_indices[b]
                # If min cover count for this block's tsubs is > 1, it's redundant
                if np.min(cover_count[tsubs]) > 1:
                    changed = True
                    cover_count[tsubs] -= 1
                else:
                    new_blocks.append(b)
            block_indices = new_blocks
            
        # 2. Swap Search (1-opt)
        # Try to replace a block b with a block b' such that:
        # - b' covers all tsubs that b uniquely covered (cover_count == 1)
        # - b' is not in the set
        # - Ideally, b' covers more or allows further reduction? 
        # For simplicity, just ensure validity.
        # This is expensive. Let's limit it.
        
        # Re-calculate cover counts for current set
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in block_indices:
            cover_count[block_tsubs_indices[b]] += 1
        
        # Identify "critical" tsubs for each block (those with cover count 1)
        # If a block has no critical tsubs, it was already removed.
        
        # Try swapping
        # For each block b in set:
        #   Let U_b = {t in b | cover_count[t] == 1}
        #   Find b' not in set such that U_b subset of b'
        #   If found, swap? 
        #   This maintains validity. Does it help reduce size? No, size is same.
        #   But it might enable further reductions?
        #   Given time constraints, simple removal is often the biggest win.
        #   Let's try a few random swaps to break local optima of the removal process?
        #   Actually, random restarts in GRASP handle that.
        
        return block_indices

    # --- Main GRASP Loop ---
    
    best_blocks = []
    best_len = float('inf')
    
    # Number of iterations depends on time
    # Each iteration: Construct (O(N_blocks * avg_updates)) + Local Search
    # Let's run as many as possible.
    
    # Adaptive alpha
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    iteration = 0
    while True:
        if check_time(5.0):
            break
            
        alpha = alphas[iteration % len(alphas)]
        
        # Random weights perturbation
        weights = base_weights * (1.0 + 0.2 * np.random.randn(num_tsubs))
        weights[weights < 0] = 1e-6
        
        cand = grasp_construct(alpha, weights)
        
        if not cand:
            continue
            
        # Check if it's a valid cover (should be, but safety)
        # And reduce
        cand = local_search(cand)
        
        if len(cand) < best_len:
            best_len = len(cand)
            best_blocks = cand
            # Print progress? No, just track.
            
        iteration += 1
        
        # If we are getting close to the theoretical lower bound or time is running out, stop?
        # Lower bound is 264. Best known 335.
        # If best_len <= 300, we are doing very well.
        
    # Final safety: if best_blocks is empty or invalid, fallback to simple greedy
    if not best_blocks:
        # Simple deterministic greedy
        uncovered = np.ones(num_tsubs, dtype=np.float64)
        selected = []
        scores = A_csr.T @ (base_weights * uncovered)
        selected_set = set()
        while True:
            if check_time(1.0): break
            b_idx = int(np.argmax(scores))
            if scores[b_idx] <= 0: break
            selected_set.add(b_idx)
            selected.append(b_idx)
            for t_idx in block_tsubs_indices[b_idx]:
                if uncovered[t_idx] > 0:
                    uncovered[t_idx] = 0
            scores = A_csr.T @ (base_weights * uncovered)
            for b in selected_set:
                scores[b] = -1e9
            if len(selected) > 500: break # Safety cap
        best_blocks = selected

    return [list(all_blocks[b]) for b in best_blocks]

