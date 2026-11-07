"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: 60s limit, safety margin 5s
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 10
    full_mask = (1 << n) - 1
    
    # Construct a standard Latin Square of order n.
    # L1[i][j] = (i + j) % n
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Helper functions for validation
    def is_latin(square):
        for i in range(n):
            if sorted(square[i]) != list(range(n)):
                return False
            col = [square[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(s1, s2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((s1[i][j], s2[i][j]))
        return len(pairs) == n*n

    def find_orthogonal_square():
        # Grid to store L2
        grid = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Bitmasks for constraints
        row_mask = [0] * n
        col_mask = [0] * n
        group_mask = [0] * n  # group_mask[k] tracks symbols used where L1[i][j] == k
        
        # List of empty cells, we will select dynamically
        # To speed up MRV, we can maintain a list of unassigned cells
        
        unassigned = [(i, j) for i in range(n) for j in range(n)]
        unassigned_idx = {(i, j): idx for idx, (i, j) in enumerate(unassigned)}
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            if not unassigned:
                return True
                
            # MRV Heuristic: Find cell with fewest candidates
            min_candidates = n + 1
            best_cell = None
            best_mask = 0
            
            # Optimization: If we don't want to scan all 100 cells every step (which is fine for 100),
            # we can just scan. 100 * 100 steps is 10k ops, trivial.
            for idx, (i, j) in enumerate(unassigned):
                g = L1[i][j]
                available = full_mask & ~(row_mask[i] | col_mask[j] | group_mask[g])
                count = bin(available).count('1')
                if count == 0:
                    return False
                if count < min_candidates:
                    min_candidates = count
                    best_cell = (i, j)
                    best_mask = available
                    if count == 1:
                        break # Can't get better than 1
            
            if best_cell is None:
                return False
                
            i, j = best_cell
            g = L1[i][j]
            
            # Get candidate values
            candidates = []
            m = best_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                candidates.append(v)
                m ^= lsb
            
            # Randomize order to avoid deterministic traps, though MRV is strong
            # random.shuffle(candidates)
            
            for v in candidates:
                bit = 1 << v
                grid[i][j] = v
                row_mask[i] |= bit
                col_mask[j] |= bit
                group_mask[g] |= bit
                
                # Remove from unassigned
                idx = unassigned_idx[(i, j)]
                last = unassigned[-1]
                unassigned_idx[last] = idx
                unassigned[idx] = last
                unassigned.pop()
                del unassigned_idx[(i, j)]
                
                if backtrack():
                    return True
                
                # Undo
                grid[i][j] = -1
                row_mask[i] &= ~bit
                col_mask[j] &= ~bit
                group_mask[g] &= ~bit
                unassigned.append((i, j))
                unassigned_idx[(i, j)] = len(unassigned) - 1
                
            return False
            
        # Fix first row to 0..9 to break symmetry? 
        # Actually, for orthogonality, we can permute symbols of L2.
        # If we fix L2[0][j] = j, we reduce the search space significantly.
        # Let's do that.
        for j in range(n):
            v = j
            bit = 1 << v
            grid[0][j] = v
            row_mask[0] |= bit
            col_mask[j] |= bit
            g = L1[0][j] # which is j
            group_mask[g] |= bit
            idx = unassigned_idx[(0, j)]
            last = unassigned[-1]
            unassigned_idx[last] = idx
            unassigned[idx] = last
            unassigned.pop()
            del unassigned_idx[(0, j)]
            
        if backtrack():
            return grid
        return None

    L2_candidate = find_orthogonal_square()
    
    if L2_candidate:
        # Verify L2 is a valid Latin square
        if is_latin(L2_candidate):
            # Verify orthogonality
            if are_orthogonal(L1, L2_candidate):
                L2 = L2_candidate
            else:
                L2 = None
        else:
            L2 = None
    else:
        L2 = None

    # Prepare result
    # Ensure types are strictly Python int and list
    L1_clean = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
    
    if L2:
        L2_clean = [[int(L2[i][j]) for j in range(n)] for i in range(n)]
        return [L1_clean, L2_clean]
    else:
        return [L1_clean]

# EVOLVE-BLOCK-END
