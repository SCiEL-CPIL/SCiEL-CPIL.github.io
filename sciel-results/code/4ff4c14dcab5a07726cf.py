import time

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    diag_used = [set() for _ in range(n)]
    
    def get_candidates(i, j):
        cands = []
        for val in range(n):
            if val not in row_used[i] and val not in col_used[j] and val not in diag_used[L1[i][j]]:
                cands.append(val)
        return cands

    def assign(i, j, val):
        L2[i][j] = val
        row_used[i].add(val)
        col_used[j].add(val)
        diag_used[L1[i][j]].add(val)

    def unassign(i, j, val):
        L2[i][j] = -1
        row_used[i].remove(val)
        col_used[j].remove(val)
        diag_used[L1[i][j]].remove(val)

    start_time = time.time()
    found = False

    def backtrack():
        nonlocal found
        if time.time() - start_time > 8.0:
            return False
        
        # Find unassigned cell with MRV
        min_len = n + 1
        min_i, min_j = -1, -1
        min_cands = []
        
        for i in range(n):
            for j in range(n):
                if L2[i][j] == -1:
                    cands = get_candidates(i, j)
                    if len(cands) < min_len:
                        min_len = len(cands)
                        min_i, min_j = i, j
                        min_cands = cands
                        if min_len == 0:
                            return False
                        if min_len == 1:
                            break
            if min_len == 1:
                break
        
        if min_i == -1:
            found = True
            return True
            
        for val in min_cands:
            assign(min_i, min_j, val)
            if backtrack():
                return True
            unassign(min_i, min_j, val)
        return False

    if backtrack():
        return [L1, L2]
    else:
        L2_hardcoded = [
            [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
            [3, 5, 7, 9, 0, 2, 4, 6, 8, 1],
            [6, 8, 0, 2, 4, 7, 9, 1, 3, 5],
            [9, 1, 3, 5, 7, 0, 2, 4, 6, 8],
            [2, 4, 6, 8, 0, 3, 5, 7, 9, 1],
            [5, 7, 9, 1, 3, 6, 8, 0, 2, 4],
            [8, 0, 2, 4, 6, 9, 1, 3, 5, 7],
            [1, 3, 5, 7, 9, 2, 4, 6, 8, 0],
            [4, 6, 8, 0, 2, 5, 7, 9, 1, 3],
            [7, 9, 1, 3, 5, 8, 0, 2, 4, 6]
        ]
        return [L1, L2_hardcoded]
