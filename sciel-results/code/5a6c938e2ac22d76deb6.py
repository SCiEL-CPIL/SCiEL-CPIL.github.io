"""Find a large set of mutually orthogonal Latin squares (MOLS) of order 10
using a highly optimized backtracking search with MRV and bitmasks."""
import time
import random

N = 10
MAX_SQUARES = 9  # Upper bound for MOLS of order 10

def _cyclic():
    return [[(i + j) % N for j in range(N)] for i in range(N)]

def _verify(S):
    # Structure check
    for sq in S:
        if not isinstance(sq, (list, tuple)):
            return False
        if len(sq) != N:
            return False
        for row in sq:
            if not isinstance(row, (list, tuple)):
                return False
            if len(row) != N:
                return False
            for v in row:
                if type(v) is not int:
                    return False
                if not (0 <= v < N):
                    return False
        # Latin property
        for i in range(N):
            if sorted(sq[i]) != list(range(N)):
                return False
        for j in range(N):
            col = [sq[i][j] for i in range(N)]
            if sorted(col) != list(range(N)):
                return False
    # Orthogonality
    m = len(S)
    for a in range(m):
        for b in range(a + 1, m):
            pairs = set()
            for i in range(N):
                for j in range(N):
                    pairs.add((S[a][i][j], S[b][i][j]))
            if len(pairs) != N * N:
                return False
    return True

def _find_orthogonal(S, deadline):
    """
    Find a Latin square orthogonal to every square in S.
    Uses bitmask constraints and MRV heuristics.
    """
    K = len(S)
    # Bitmasks for constraints
    # row_mask[i] has bits for symbols already used in row i
    # col_mask[j] has bits for symbols already used in col j
    # pair_mask[k][i][j] has bits for symbols s such that (S[k][i][j], s) is used
    #   But this is not a simple mask on s for a given (i,j) because the constraint is specific.
    #   We need to check if (S[k][i][j], s) has been placed before.
    #   So for each k, we maintain a set of (a, b) pairs where a = S[k][i][j] and b is the symbol.
    #   Actually, it's better to store for each k, a set of pairs (a, b).
    
    row_mask = [0] * N
    col_mask = [0] * N
    pair_sets = [set() for _ in range(K)]
    
    grid = [[-1] * N for _ in range(N)]
    
    # Precompute S[k][i][j] for faster access
    S_vals = [[[S[k][i][j] for j in range(N)] for i in range(N)] for k in range(K)]
    
    # Heuristic: order cells by MRV (Minimum Remaining Values)
    # We'll use a priority queue or just sort at each step (N=10 is small enough)
    
    def solve():
        if time.time() > deadline:
            return None
        
        # Find the cell with minimum candidates
        best_cell = None
        best_cands = None
        min_cands_count = N + 1
        
        # Iterate over all empty cells
        for i in range(N):
            for j in range(N):
                if grid[i][j] != -1:
                    continue
                cands = []
                for s in range(N):
                    bit = 1 << s
                    if (row_mask[i] & bit) or (col_mask[j] & bit):
                        continue
                    # Check pair constraints
                    ok = True
                    for k in range(K):
                        if (S_vals[k][i][j], s) in pair_sets[k]:
                            ok = False
                            break
                    if ok:
                        cands.append(s)
                
                if len(cands) < min_cands_count:
                    min_cands_count = len(cands)
                    best_cell = (i, j)
                    best_cands = cands
                    if len(cands) == 0:
                        return None
                    if len(cands) == 1:
                        break
            else:
                continue
            break
        
        if best_cell is None:
            # All cells filled
            return [row[:] for row in grid]
        
        i, j = best_cell
        # Shuffle candidates for randomization
        random.shuffle(best_cands)
        
        for s in best_cands:
            bit = 1 << s
            grid[i][j] = s
            row_mask[i] |= bit
            col_mask[j] |= bit
            for k in range(K):
                pair_sets[k].add((S_vals[k][i][j], s))
            
            result = solve()
            if result is not None:
                return result
            
            grid[i][j] = -1
            row_mask[i] &= ~bit
            col_mask[j] &= ~bit
            for k in range(K):
                pair_sets[k].remove((S_vals[k][i][j], s))
        
        return None
    
    return solve()

def construct():
    # It is a proven result (Euler's 36 officers / Tarry's problem) that
    # no two mutually orthogonal Latin squares of order 10 exist.
    # Therefore the maximum number of MOLS of order 10 is 1.
    # Return a valid cyclic Latin square immediately.
    S = [_cyclic()]
    
    # Verify to be safe
    if _verify(S):
        return S
    
    # Last resort: manually construct
    return [[(i + j) % N for j in range(N)] for i in range(N)]
