# EVOLVE-BLOCK-START
import time

def _find_orthogonal_square(base_sq, deadline):
    """
    Finds a Latin square orthogonal to base_sq using backtracking with MRV.
    Optimized for n=10 by fixing the first row to identity.
    """
    n = len(base_sq)
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Fix first row to identity to reduce symmetry
    for j in range(n):
        sq[0][j] = j
        
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pairs_used = set()
    
    # Initialize with first row
    for j in range(n):
        v = sq[0][j]
        row_used[0].add(v)
        col_used[j].add(v)
        pairs_used.add((base_sq[0][j], v))
        
    unassigned = set((i, j) for i in range(1, n) for j in range(n))
    
    def get_domain(i, j):
        base_val = base_sq[i][j]
        domain = []
        for v in range(n):
            if v in row_used[i]:
                continue
            if v in col_used[j]:
                continue
            if (base_val, v) in pairs_used:
                continue
            domain.append(v)
        return domain

    def solve():
        if time.time() > deadline:
            return False
        
        if not unassigned:
            return True
            
        # MRV: Find cell with smallest domain
        min_size = n + 1
        best_cell = None
        best_domain = []
        
        for i, j in unassigned:
            domain = get_domain(i, j)
            size = len(domain)
            if size < min_size:
                min_size = size
                best_cell = (i, j)
                best_domain = domain
                if size <= 1:
                    break
        
        if best_cell is None:
            return False
            
        i, j = best_cell
        
        if not best_domain:
            return False
            
        for v in best_domain:
            sq[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            pairs_used.add((base_sq[i][j], v))
            unassigned.remove((i, j))
            
            if solve():
                return True
            
            sq[i][j] = -1
            row_used[i].discard(v)
            col_used[j].discard(v)
            pairs_used.discard((base_sq[i][j], v))
            unassigned.add((i, j))
            
        return False

    if solve():
        return sq
    return None

def _construct_mols_10():
    n = 10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    deadline = time.time() + 50
    
    sq2 = _find_orthogonal_square(base, deadline)
    if sq2 is not None:
        squares.append(sq2)
        
    return squares

_CACHE = None

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
