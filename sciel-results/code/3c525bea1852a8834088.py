import time

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _find_partner(L1, deadline):
    n = 10
    L2 = [[None for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    used_pairs = set()
    
    # Pre-compute for each cell (i, j) the set of values that are NOT in L1[i][j]
    # Actually, we need (L1[i][j], v) not in used_pairs.
    # We can optimize the availability check.
    
    # Heuristic: order cells by number of available values (MRV - Minimum Remaining Values)
    # But since the structure is highly constrained, simple row-major might be fine if we have good pruning.
    # Let's try a slightly smarter ordering: process cells in an order that might fill constraints faster.
    # Or just stick to row-major but with fast checks.
    
    # A better heuristic for backtracking: choose the next cell with the fewest available candidates.
    # This is expensive to compute dynamically for every node.
    # Given n=10, row-major is often sufficient if the first square is "nice".
    # However, L1 = (i+j)%10 is a group table.
    
    # Let's try to optimize the inner loop.
    def bt(i, j):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        if j == n:
            return bt(i+1, 0)
            
        # Available values
        # v must be:
        # 1. not in row_used[i]
        # 2. not in col_used[j]
        # 3. (L1[i][j], v) not in used_pairs
        
        # We can iterate through 0..9 and check
        avail = []
        l1_val = L1[i][j]
        for v in range(n):
            if v in row_used[i]:
                continue
            if v in col_used[j]:
                continue
            if (l1_val, v) in used_pairs:
                continue
            avail.append(v)
        
        if not avail:
            return False
            
        # Randomize order? Or keep deterministic?
        # Deterministic is safer for reproducibility, but randomization might help escape local minima if we were doing something else.
        # Here, backtracking is complete, so order only affects speed.
        
        for v in avail:
            L2[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            used_pairs.add((l1_val, v))
            
            if bt(i, j+1):
                return True
            
            L2[i][j] = None
            row_used[i].remove(v)
            col_used[j].remove(v)
            used_pairs.remove((l1_val, v))
            
        return False

    if bt(0, 0):
        return L2
    return None

def construct():
    deadline = time.time() + 50
    n = 10
    
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    partner = _find_partner(sq1, deadline)
    
    if partner:
        # Verify
        if _is_latin(partner) and _orthogonal(sq1, partner):
            return [sq1, partner]
            
    return [sq1]
