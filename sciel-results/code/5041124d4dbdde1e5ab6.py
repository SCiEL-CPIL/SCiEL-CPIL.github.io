import sys
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) using the relationship with S(4,5,11).
    The large Witt design S(5,6,12) on {0..11} can be constructed from the small Witt design S(4,5,11) on {0..10} as follows:
    Let W11 be the set of 66 blocks of S(4,5,11).
    The blocks of S(5,6,12) are:
    1. B U {11} for each B in W11
    2. (V \ B) for each B in W11, where V = {0..10}
    """
    
    # 1. Solve for S(4,5,11)
    points_11 = range(11)
    all_4subsets = list(combinations(points_11, 4))
    col_index_4 = {s: i for i, s in enumerate(all_4subsets)}
    num_cols_4 = len(all_4subsets) # 330
    
    all_5subsets_11 = list(combinations(points_11, 5))
    rows_11 = []
    for b in all_5subsets_11:
        cols = [col_index_4[s] for s in combinations(b, 4)]
        rows_11.append(cols)
        
    col_to_rows_4 = [[] for _ in range(num_cols_4)]
    for r_idx, cols in enumerate(rows_11):
        for c in cols:
            col_to_rows_4[c].append(r_idx)
            
    solution_11 = []
    used_rows_11 = set()
    covered_4 = set()
    
    def search_11(depth):
        if len(covered_4) == num_cols_4:
            return True
            
        min_count = float('inf')
        best_col = -1
        best_candidates = []
        
        for c in range(num_cols_4):
            if c in covered_4:
                continue
                
            candidates = []
            for r in col_to_rows_4[c]:
                if r in used_rows_11:
                    continue
                conflict = False
                for cc in rows_11[r]:
                    if cc in covered_4 and cc != c:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(r)
            
            if len(candidates) < min_count:
                min_count = len(candidates)
                best_col = c
                best_candidates = candidates
                
                if min_count == 0:
                    return False
                if min_count == 1:
                    break
            
        if best_col == -1:
            return False
            
        for r in best_candidates:
            used_rows_11.add(r)
            new_covered_cols = [cc for cc in rows_11[r] if cc not in covered_4]
            for cc in new_covered_cols:
                covered_4.add(cc)
            
            solution_11.append(r)
            
            if search_11(depth + 1):
                return True
            
            solution_11.pop()
            used_rows_11.remove(r)
            for cc in new_covered_cols:
                covered_4.remove(cc)
                
        return False

    sys.setrecursionlimit(10000)
    
    if not search_11(0):
        return []

    # 2. Construct S(5,6,12) from S(4,5,11)
    result = []
    V = set(range(11))
    for r_idx in solution_11:
        B = set(all_5subsets_11[r_idx])
        # Block 1: B U {11}
        result.append(sorted(list(B) + [11]))
        # Block 2: V \ B
        comp = V - B
        result.append(sorted(list(comp)))
        
    return result
