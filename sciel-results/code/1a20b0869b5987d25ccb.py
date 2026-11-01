import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    D = 11
    
    # Generate weight-5 vectors: 5 positions with ±1, rest 0
    # C(11,5) = 462 supports, 2^5 = 32 signs each
    # Total: 462 * 32 = 14784 vectors
    supports = list(itertools.combinations(range(D), 5))
    n_supports = len(supports)  # 462
    
    all_vectors = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 14784
    
    # Build adjacency list for conflicts (dot > 2)
    # Two weight-5 vectors conflict if their dot product > 2
    # Dot product = sum of products over shared positions
    # Max shared positions = 5 (if same support)
    # If same support: dot ranges from -5 to 5, conflict if dot > 2, i.e., dot >= 3
    # If sharing 4 positions: dot ranges from -4 to 4, conflict if dot > 2, i.e., dot >= 3
    # If sharing 3 positions: dot ranges from -3 to 3, conflict if dot > 2, i.e., dot == 3
    # If sharing 2 or fewer: dot <= 2, no conflict
    
    # Build tri_to_pent: for each 3-subset, which supports contain it
    tri_to_pent = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_pent[tri].append(i)
    
    # Also need to handle same-support conflicts (sharing 5 positions)
    # and 4-shared-position conflicts
    
    # For efficiency, let's use the full dot product matrix approach but in blocks
    # Actually, let's build adjacency by iterating over pairs of supports that share >= 3 elements
    
    adj = [[] for _ in range(n_total)]
    
    # Process same-support conflicts first (share 5 positions)
    for s_idx in range(n_supports):
        base = s_idx * 32
        v_block = all_vectors_np[base:base+32]
        d = v_block @ v_block.T
        # Conflict if d > 2 (excluding diagonal)
        indices = np.argwhere(d > 2)
        for i, j in indices:
            if i != j:
                idx1 = base + i
                idx2 = base + j
                if idx2 not in adj[idx1]:
                    adj[idx1].append(idx2)
                    adj[idx2].append(idx1)
    
    # Process different-support conflicts (share 4 or 3 positions)
    # Use tet_to_pent for 4-subsets
    tet_to_pent = defaultdict(list)
    for i, s in enumerate(supports):
        for tet in itertools.combinations(s, 4):
            tet_to_pent[tet].append(i)
    
    # For pairs of supports sharing exactly 4 elements
    for tet, pents in tet_to_pent.items():
        for u in range(len(pents)):
            for v in range(u + 1, len(pents)):
                s1 = pents[u]
                s2 = pents[v]
                base1 = s1 * 32
                base2 = s2 * 32
                v1 = all_vectors_np[base1:base1+32]
                v2 = all_vectors_np[base2:base2+32]
                d = v1 @ v2.T
                indices = np.argwhere(d > 2)
                for i, j in indices:
                    idx1 = base1 + i
                    idx2 = base2 + j
                    if idx2 not in adj[idx1]:
                        adj[idx1].append(idx2)
                        adj[idx2].append(idx1)
    
    # For pairs of supports sharing exactly 3 elements
    for tri, pents in tri_to_pent.items():
        for u in range(len(pents)):
            for v in range(u + 1, len(pents)):
                s1 = pents[u]
                s2 = pents[v]
                # Check if they share exactly 3 (not 4 or 5)
                shared = set(supports[s1]) & set(supports[s2])
                if len(shared) != 3:
                    continue
                base1 = s1 * 32
                base2 = s2 * 32
                v1 = all_vectors_np[base1:base1+32]
                v2 = all_vectors_np[base2:base2+32]
                d = v1 @ v2.T
                indices = np.argwhere(d > 2)
                for i, j in indices:
                    idx1 = base1 + i
                    idx2 = base2 + j
                    if idx2 not in adj[idx1]:
                        adj[idx1].append(idx2)
                        adj[idx2].append(idx1)
    
    # Initial degrees
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    # Greedy Minimum Degree with randomization + local improvement
    max_restarts = 200
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n_total, dtype=bool)
        
        # Greedy Min-Degree
        while avail.any():
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break
            
            degs = deg[avail_indices]
            min_deg = np.min(degs)
            min_indices_rel = np.where(degs == min_deg)[0]
            
            # Random tie-breaking
            pick_rel = min_indices_rel[rng.integers(len(min_indices_rel))]
            u = avail_indices[pick_rel]
            
            selected[u] = True
            avail[u] = False
            
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
        
        k = np.sum(selected)
        
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
    
    # Local improvement: try to add vertices or swap
    if best_selected is not None:
        # Try adding non-selected vertices that don't conflict with selected ones
        for i in range(n_total):
            if not best_selected[i]:
                # Check if i conflicts with any selected vertex
                conflict = False
                for v in adj[i]:
                    if best_selected[v]:
                        conflict = True
                        break
                if not conflict:
                    best_selected[i] = True
        
        # Try swaps: remove one selected, add two non-selected
        for i in range(n_total):
            if best_selected[i]:
                # Try removing i and adding neighbors of i that are non-selected
                # This is complex; skip for now to save time
                pass
    
    # Recount
    best_k = np.sum(best_selected)
    
    result_w5 = all_vectors_np[best_selected].astype(np.int64)
    
    # Add weight-1 vectors (norm 2, entries ±2)
    # Check compatibility with weight-5 vectors (norm sqrt(5))
    # R^2 = 5
    # ||w1 - u5||^2 >= 5
    # 4 + 5 - 2*w1.u5 >= 5
    # w1.u5 <= 2
    # w1 = ±2e_i, u5 has ±1 at position i
    # w1.u5 = ±2*(±1) = ±2
    # Max is 2, so condition w1.u5 <= 2 is satisfied.
    # Weight-1 vectors compatible with each other:
    # ||v1 - v2||^2 >= 5
    # 4 + 4 - 2*v1.v2 >= 5
    # v1.v2 <= 1.5, so v1.v2 <= 1 (integer)
    # Same pos same sign: dot = 4 > 1. Conflict.
    # Same pos diff sign: dot = -4. OK.
    # Diff pos: dot = 0. OK.
    # So we can take at most 1 per coordinate, but both signs of same coordinate are OK.
    # Take all 22.
    
    weight1_vectors = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            weight1_vectors.append(v)
    
    weight1_np = np.array(weight1_vectors, dtype=np.int64)
    
    final_vectors = np.vstack([result_w5, weight1_np])
    
    return final_vectors
