import time
import random

def construct():
    n = 10
    # L1 is the standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We want to find L2 such that L1 and L2 are orthogonal.
    # This is equivalent to finding a Latin Square L2 such that 
    # for each a in 0..9, the values L2[i][j] where L1[i][j]==a are distinct.
    # Since L1[i][j] == a iff (i+j)%n == a, this means 
    # the values on each wrapping diagonal must be distinct.
    
    # Backtracking search for L2
    # State: grid, row_masks, col_masks, diag_masks
    # masks are integers where bit k is set if symbol k is used.
    
    grid = [[None for _ in range(n)] for _ in range(n)]
    row_masks = [0] * n
    col_masks = [0] * n
    diag_masks = [0] * n
    
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # Precompute cells for MRV heuristic? 
    # Or just iterate. With N=10, even simple backtracking might be fast.
    # But MRV is safer.
    
    # We will fill cells in a specific order to help propagation.
    # Actually, randomization + MRV is robust.
    
    def is_valid(i, j, val):
        mask = 1 << val
        if row_masks[i] & mask: return False
        if col_masks[j] & mask: return False
        d = (i + j) % n
        if diag_masks[d] & mask: return False
        return True

    def solve(idx):
        if time.time() > deadline:
            return False
            
        if idx == n * n:
            return True
            
        # Find the cell with Minimum Remaining Values (MRV)
        # among the unassigned cells from idx onwards?
        # Simpler: just pick the next unassigned cell in row-major order.
        # For N=10, this might be slow if not guided.
        # Let's use a simple heuristic: pick the first unassigned cell.
        
        # Actually, let's just iterate through all cells to find the best candidate.
        # But that's expensive.
        # Let's stick to row-major order but with value ordering.
        
        # Find next empty cell
        r, c = divmod(idx, n)
        
        # If this cell is already filled (shouldn't be in this loop structure), skip?
        # No, we fill sequentially.
        
        # Try values 0..9
        # Shuffle values to avoid bias?
        vals = list(range(n))
        random.shuffle(vals)
        
        for v in vals:
            if is_valid(r, c, v):
                grid[r][c] = v
                mask = 1 << v
                row_masks[r] |= mask
                col_masks[c] |= mask
                d = (r + c) % n
                diag_masks[d] |= mask
                
                if solve(idx + 1):
                    return True
                
                # Undo
                grid[r][c] = None
                row_masks[r] ^= mask
                col_masks[c] ^= mask
                diag_masks[d] ^= mask
                
        return False

    # To make it faster, we can use a better cell selection.
    # Let's try a simple backtracking first. If it's too slow, we might need to optimize.
    # Given N=10, the search space is large, but constraints are tight.
    # Diagonal constraints are very strong.
    
    # Let's add a deadline check inside the loop.
    
    # I will use a slightly more efficient solver structure if needed, 
    # but let's try this simple one first. 
    # If it times out, I return 1 square.
    
    # Actually, let's use a known solution if possible to avoid search risk.
    # But I don't have one hardcoded.
    
    # Let's try to run the solver.
    # To increase chance of success, I'll use a random seed and try a few times?
    # No, just one run with good heuristics.
    
    # Better heuristic: Forward Checking + MRV.
    # I'll implement a simple MRV selection.
    
    def solve_mrv():
        if time.time() > deadline:
            return False
            
        # Find cell with min candidates
        min_candidates = n + 1
        best_r, best_c = -1, -1
        best_vals = []
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] is not None:
                    continue
                avail = []
                for v in range(n):
                    if is_valid(i, j, v):
                        avail.append(v)
                if len(avail) < min_candidates:
                    min_candidates = len(avail)
                    best_r, best_c = i, j
                    best_vals = avail
                    if min_candidates == 0:
                        return False
                    if min_candidates == 1:
                        break
            if min_candidates == 1:
                break
        
        if best_r == -1:
            # All filled
            return True
            
        # Shuffle best_vals to break symmetry?
        random.shuffle(best_vals)
        
        for v in best_vals:
            grid[best_r][best_c] = v
            mask = 1 << v
            row_masks[best_r] |= mask
            col_masks[best_c] |= mask
            d = (best_r + best_c) % n
            diag_masks[d] |= mask
            
            if solve_mrv():
                return True
            
            grid[best_r][best_c] = None
            row_masks[best_r] ^= mask
            col_masks[best_c] ^= mask
            diag_masks[d] ^= mask
            
        return False

    # Run solver
    random.seed(42) # Deterministic for reproducibility in evaluation?
    # Actually, random seed might be better to explore different paths if first fails?
    # But MRV is deterministic given the grid state.
    
    success = solve_mrv()
    
    if success:
        # Verify L2 is a Latin Square and orthogonal to L1
        # L2 is guaranteed to be Latin by row/col masks.
        # Orthogonality is guaranteed by diag masks.
        return [L1, [[grid[i][j] for j in range(n)] for i in range(n)]]
    else:
        # Fallback
        return [L1]
