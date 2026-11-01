import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Candidate norms squared
    # Norm 2: 220 vectors (baseline)
    # Norm 5: 440 vectors
    # Norm 10: 440 vectors
    # Norm 13: 440 vectors (3,2)
    # Norm 17: 440 vectors (4,1)
    
    best_vectors = None
    best_k = 0
    
    # 1. Generate baseline (Norm 2)
    v_norm2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                v_norm2.append(v)
    best_k = len(v_norm2)
    best_vectors = v_norm2
    
    # Helper to generate candidates for a given norm squared
    # We look for integer solutions to sum(x_i^2) = S
    # For small S, we can just iterate.
    
    def get_candidates(S):
        cands = []
        # Iterate over possible non-zero components
        # Max component is sqrt(S)
        max_comp = int(np.sqrt(S)) + 1
        
        # Simple brute force for small D and S?
        # Better: Generate all combinations of indices and values.
        # Since S is small (<= 20), non-zero components are few.
        
        # We can use recursion or itertools.
        # Let's just generate all vectors with L2 norm squared S.
        # To avoid duplicates, we can enforce some ordering or just use a set.
        
        seen = set()
        
        # Max number of non-zero elements
        # 1^2 + ... 
        # For S=5, (2,1) or (1,2)
        # For S=10, (3,1)
        # For S=13, (3,2)
        # For S=17, (4,1)
        # For S=20, (4,2) or (2,2,2,2,2,2,2)? No.
        
        # We'll just generate all integer vectors with norm squared S.
        # Efficient way:
        # Iterate over number of non-zeros k.
        for k in range(1, D+1):
            # Min sum of squares for k non-zeros is k (all 1s).
            # If k > S, break.
            if k > S:
                break
            
            # Generate all ways to put k non-zeros
            # This is expensive if done naively.
            # But S is small.
            
            pass
            
        # Alternative: Just check a predefined list of "shapes"
        # Shapes for S=5: (2,1)
        # Shapes for S=10: (3,1)
        # Shapes for S=13: (3,2)
        # Shapes for S=17: (4,1)
        # Shapes for S=20: (4,2), (2,2,2,2,2,2,2,2,2,2,2) no.
        
        # Let's hardcode the shapes for efficiency and safety.
        shapes = []
        # Find integer partitions of S into squares
        # e.g. S=5: 4+1
        # S=10: 9+1
        # S=13: 9+4
        # S=17: 16+1
        # S=20: 16+4, 4+4+4+4+4+4+4+4+4+4+4 (no)
        
        # We'll just iterate through all possible vectors in a bounded box?
        # Box size sqrt(S).
        # For S=5, box is [-2, 2]. 5^11 is too big.
        
        # Back to shapes:
        # Function to get all sorted tuples of squared values summing to S
        def get_sq_shapes(S):
            res = []
            def rec(rem, max_val, curr):
                if rem == 0:
                    if curr:
                        res.append(tuple(curr))
                    return
                for v in range(max_val, 0, -1):
                    sq = v*v
                    if sq > rem:
                        continue
                    curr.append(sq)
                    rec(rem - sq, v, curr)
                    curr.pop()
            rec(S, int(np.sqrt(S)), [])
            return res
        
        sq_shapes = get_sq_shapes(S)
        
        for shape in sq_shapes:
            # shape is a tuple of squared values, e.g. (4, 1)
            # Values are sqrt(shape)
            vals = [int(np.sqrt(x)) for x in shape]
            # Permute indices and signs
            
            # Number of non-zeros
            nz = len(vals)
            
            # Choose indices
            for idxs in itertools.combinations(range(D), nz):
                # Permute values
                for perm in set(itertools.permutations(vals)):
                    # Signs
                    # To avoid duplicates, we can fix the sign of the first non-zero to be positive?
                    # No, we need all sign combinations.
                    # But (2,1) and (-2,-1) are distinct.
                    # (2,1) and (2,-1) are distinct.
                    
                    # Iterate over all 2^nz sign combinations
                    for signs in itertools.product([1, -1], repeat=nz):
                        v = [0] * D
                        for k in range(nz):
                            v[idxs[k]] = perm[k] * signs[k]
                        # Add to candidates
                        cands.append(v)
                        
        # Remove duplicates
        unique_cands = []
        seen = set()
        for v in cands:
            t = tuple(v)
            if t not in seen:
                seen.add(t)
                unique_cands.append(v)
                
        return unique_cands

    def solve_independent_set(candidates, S, time_limit=10.0):
        n = len(candidates)
        if n == 0:
            return []
        
        # Precompute distances squared
        # dist[i][j]
        # n is ~440. 440*440 = 193600.
        # Store as list of lists or numpy array.
        dists = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            vi = candidates[i]
            for j in range(i+1, n):
                vj = candidates[j]
                d2 = sum((a-b)**2 for a, b in zip(vi, vj))
                dists[i][j] = d2
                dists[j][i] = d2
                
        # Conflict if dists[i][j] < S
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if dists[i][j] < S:
                    adj[i].append(j)
                    adj[j].append(i)
                    
        # Greedy initialization
        # Sort by degree
        degrees = [len(a) for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x])
        
        selected = set()
        removed = [False] * n
        
        for i in order:
            if removed[i]:
                continue
            # Check if i conflicts with any selected
            valid = True
            for j in adj[i]:
                if j in selected:
                    valid = False
                    break
            if valid:
                selected.add(i)
                # Mark neighbors as removed? 
                # Actually, we just skip them if they conflict with selected.
                # This greedy is "pick min degree, add if possible".
                
        # Local Search / SA
        current_set = list(selected)
        current_size = len(current_set)
        
        # Convert to set for fast lookup
        sel_set = set(current_set)
        
        def is_valid_add(v):
            for u in sel_set:
                if dists[v][u] < S:
                    return False
            return True
            
        def is_valid_swap(v_in, v_out):
            # Remove v_in, add v_out
            # Check v_out against (sel_set - {v_in})
            for u in sel_set:
                if u == v_in:
                    continue
                if dists[v_out][u] < S:
                    return False
            return True
            
        best_set = list(sel_set)
        best_size = current_size
        
        start = time.time()
        temp = 10.0
        cooling = 0.99
        
        while time.time() < time_limit + start:
            if not current_set:
                break
                
            # Random move
            move_type = np.random.rand()
            
            if move_type < 0.4:
                # Try to add
                # Pick random node not in set
                candidates_to_add = [i for i in range(n) if i not in sel_set]
                if candidates_to_add:
                    v = np.random.choice(candidates_to_add)
                    if is_valid_add(v):
                        sel_set.add(v)
                        current_set.append(v)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = list(sel_set)
                    # If not valid, do nothing (or penalty)
                    
            elif move_type < 0.7:
                # Try to remove
                v = np.random.choice(current_set)
                sel_set.remove(v)
                current_set.remove(v)
                current_size -= 1
                
            else:
                # Try to swap
                v_in = np.random.choice(current_set)
                candidates_to_add = [i for i in range(n) if i not in sel_set]
                if candidates_to_add:
                    v_out = np.random.choice(candidates_to_add)
                    if is_valid_swap(v_in, v_out):
                        sel_set.remove(v_in)
                        current_set.remove(v_in)
                        sel_set.add(v_out)
                        current_set.append(v_out)
                        # Size remains same
                        
            temp *= cooling
            
        return [candidates[i] for i in best_set]

    # Try Norm 5
    if time.time() < deadline:
        cands_5 = get_candidates(5)
        sol_5 = solve_independent_set(cands_5, 5, time_limit=20.0)
        if len(sol_5) > best_k:
            best_k = len(sol_5)
            best_vectors = sol_5
            
    # Try Norm 10
    if time.time() < deadline:
        cands_10 = get_candidates(10)
        sol_10 = solve_independent_set(cands_10, 10, time_limit=20.0)
        if len(sol_10) > best_k:
            best_k = len(sol_10)
            best_vectors = sol_10
            
    # Try Norm 13
    if time.time() < deadline:
        cands_13 = get_candidates(13)
        sol_13 = solve_independent_set(cands_13, 13, time_limit=20.0)
        if len(sol_13) > best_k:
            best_k = len(sol_13)
            best_vectors = sol_13
            
    # Try Norm 17
    if time.time() < deadline:
        cands_17 = get_candidates(17)
        sol_17 = solve_independent_set(cands_17, 17, time_limit=20.0)
        if len(sol_17) > best_k:
            best_k = len(sol_17)
            best_vectors = sol_17

    return np.array(best_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
