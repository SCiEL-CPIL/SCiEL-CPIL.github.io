import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0
    t0 = time.time()
    
    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Bitset size
    n_words = (n_tsubsets + 63) // 64
    
    # block_tsubs: list of arrays of t-subset indices for each block
    block_tsubs = [None] * n_blocks
    # block_bitsets: numpy array (n_blocks, n_words) of uint64
    block_bitsets = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    rows = []
    cols = []
    data = []
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = np.array(tsubs, dtype=np.int32)
        
        # Set bits in block_bitsets[bi]
        for ts_idx in tsubs:
            word_idx = ts_idx // 64
            bit_idx = ts_idx % 64
            block_bitsets[bi, word_idx] |= (1 << bit_idx)
            
            rows.append(ts_idx)
            cols.append(bi)
            data.append(1.0)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
    
    # tsub_blocks: list of arrays of block indices for each t-subset
    tsub_blocks = [None] * n_tsubsets
    for i in range(n_tsubsets):
        tsub_blocks[i] = M[i, :].indices.astype(np.int32)
        
    # Helper: Convert cover_count array to bitset
    def cover_count_to_bitset(cc):
        # cc is np array of int
        mask = cc >= 2
        # Reshape and pack
        # Pad mask to multiple of 64
        pad_len = (64 - (len(mask) % 64)) % 64
        mask_padded = np.concatenate([mask, np.zeros(pad_len, dtype=bool)])
        mask_res = mask_padded.reshape(-1, 64)
        
        # Convert each row of 64 bools to a uint64
        # Weights: 1, 2, 4, ..., 2^63
        weights = np.array([1 << i for i in range(64)], dtype=np.uint64)
        # Sum of (mask_res * weights)
        # Note: bool * uint64 works in numpy?
        # Let's use .astype(np.uint64)
        mask_u = mask_res.astype(np.uint64)
        bitset = mask_u @ weights
        return bitset

    # Helper: Check if A is subset of B (bitsets)
    def is_subset(A, B):
        # A & B == A
        return np.all((A & B) == A)
        
    # Helper: Get set bits of a bitset
    def get_set_bits(bitset):
        bits = []
        for w in range(len(bitset)):
            word = int(bitset[w])
            while word:
                # Find index of least significant set bit
                # (word & -word) isolates the lowest set bit
                # .bit_length() - 1 gives the index
                bit = (word & -word).bit_length() - 1
                bits.append(w * 64 + bit)
                word &= word - 1
        return bits

    # 2. LP Relaxation
    dual_weights = None
    try:
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            dual_weights = np.abs(res.ineqlin.marginals)
            if np.sum(dual_weights) == 0:
                dual_weights = np.ones(n_tsubsets)
    except:
        dual_weights = np.ones(n_tsubsets)

    # 3. Greedy Initialization
    def lp_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        block_weights = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            block_weights[bi] = np.sum(dual_weights[block_tsubs[bi]])
            
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
                
            max_w = block_weights.max()
            if max_w <= 1e-9:
                break
                
            chosen = np.argmax(block_weights)
            selected_blocks.append(chosen)
            
            ts_list = block_tsubs[chosen]
            active_mask = uncovered[ts_list]
            
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                uncovered[active_ts_indices] = False
                
                for ts_idx in active_ts_indices:
                    w_sub = dual_weights[ts_idx]
                    if w_sub == 0:
                        continue
                    blocks_to_update = tsub_blocks[ts_idx]
                    block_weights[blocks_to_update] -= w_sub
                    
            block_weights[chosen] = -1e10
            
        return selected_blocks

    # 4. Local Search with Bitsets
    def local_search(blocks, seed=0, max_iterations=500):
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        reduced = list(blocks)
        best_reduced = list(reduced)
        best_len = len(reduced)
        
        # State
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for b in reduced:
            for idx in block_tsubs[b]:
                cover_count[idx] += 1
                
        double_cover = cover_count_to_bitset(cover_count)
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
                
            improved = False
            
            # Phase 1: Remove redundant blocks
            # We iterate through blocks. If a block is redundant, remove it.
            # We need to recompute double_cover after removal.
            # To be efficient, we can try to remove all redundant blocks in one pass,
            # but removing one changes cover_count, so double_cover changes.
            # However, if block i is redundant, removing it only *decreases* cover_count.
            # So a block that was NOT redundant might become redundant? No.
            # A block becomes redundant if all its tsubs are covered by others.
            # Removing a block makes it harder for others to be redundant.
            # So if we scan and remove, we might miss some?
            # Actually, if i is redundant, removing it doesn't make j redundant.
            # It might make j *less* likely to be redundant.
            # So a single pass is sufficient to find *all* currently redundant blocks?
            # Yes, because redundancy is a property of the current set.
            # If i is redundant in S, and we remove i to get S\{i},
            # any block j in S\{i} that was redundant in S is still redundant in S\{i}?
            # j redundant in S means S_j \subseteq \cup_{k \in S\{j}} S_k.
            # In S\{i}, the union is \cup_{k \in S\{j,i}} S_k.
            # This is a subset of \cup_{k \in S\{j}} S_k.
            # So S_j might NOT be a subset of the smaller union.
            # So j might become NON-redundant.
            # Therefore, we must re-evaluate after each removal.
            
            changed = True
            while changed:
                changed = False
                # Recompute double_cover
                double_cover = cover_count_to_bitset(cover_count)
                
                for i in range(len(reduced)):
                    if time.time() > deadline - 2:
                        break
                    
                    # Compute U_i = S_i \ double_cover
                    # S_i is block_bitsets[reduced[i]]
                    # U_i is bitset
                    U_i = block_bitsets[reduced[i]] & ~double_cover
                    # Check if U_i is empty
                    if not np.any(U_i):
                        # Remove block i
                        b_idx = reduced[i]
                        for idx in block_tsubs[b_idx]:
                            cover_count[idx] -= 1
                        reduced.pop(i)
                        changed = True
                        improved = True
                        if len(reduced) < best_len:
                            best_len = len(reduced)
                            best_reduced = list(reduced)
                        break # Restart scan
                
            if len(reduced) < 2:
                break
                
            # Phase 2: Swap
            if not improved:
                double_cover = cover_count_to_bitset(cover_count)
                n = len(reduced)
                order = list(range(n))
                rng.shuffle(order)
                
                for i in order:
                    if time.time() > deadline - 2:
                        break
                        
                    U_i = block_bitsets[reduced[i]] & ~double_cover
                    if not np.any(U_i):
                        # Should have been caught in Phase 1, but maybe due to timing?
                        # Just skip
                        continue
                        
                    # Get a random t-subset from U_i
                    bits = get_set_bits(U_i)
                    if not bits:
                        continue
                    u = bits[rng.randint(0, len(bits)-1)]
                    
                    # Find candidates
                    candidates = tsub_blocks[u]
                    found_swap = False
                    rng.shuffle(candidates)
                    
                    for ci in candidates:
                        if ci in reduced:
                            continue
                        
                        # Check if S_ci covers U_i
                        if is_subset(U_i, block_bitsets[ci]):
                            # Perform swap
                            old_b = reduced[i]
                            for idx in block_tsubs[old_b]:
                                cover_count[idx] -= 1
                            for idx in block_tsubs[ci]:
                                cover_count[idx] += 1
                            reduced[i] = ci
                            improved = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        break # Restart Phase 1
                
            # Cooling
            T *= cooling_rate
            
            if not improved and T <= T_min:
                break
                
        return best_reduced

    best_blocks = None
    best_len = float('inf')
    
    num_trials = 30
    
    for seed in range(num_trials):
        if time.time() > deadline - 15:
            break
            
        blocks_idx = lp_greedy(seed)
        if not blocks_idx:
            continue
            
        # Local search
        blocks_idx = local_search(blocks_idx, seed, max_iterations=200)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = list(blocks_idx)
            
    # Final polish
    if best_blocks is not None and time.time() < deadline - 10:
        final = local_search(best_blocks, 999, max_iterations=500)
        if len(final) < best_len:
            best_blocks = final
            best_len = len(final)
            
    if best_blocks is None:
        # Fallback
        blocks_idx = lp_greedy(0)
        best_blocks = local_search(blocks_idx, 0, max_iterations=100)
        
    return [all_blocks[i] for i in best_blocks]
