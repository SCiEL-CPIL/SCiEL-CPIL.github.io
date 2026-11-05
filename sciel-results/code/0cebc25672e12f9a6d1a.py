import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def is_valid_solution(M, A):
        if len(A) > 32:
            return False
        if 0 not in A:
            return False
        if max(A) > M:
            return False
        
        diffs = set()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False
                diffs.add(d)
        
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def dfs(A, M, current_diffs, missing_diffs):
        if time.time() > deadline:
            return None
            
        if not missing_diffs:
            return A
        
        if len(A) >= 32:
            return None
            
        m_max = 32 - len(A)
        k_curr = len(A)
        max_new_diffs = m_max * k_curr + m_max * (m_max - 1) // 2
        if len(missing_diffs) > max_new_diffs:
            return None

        candidates = []
        for x in range(1, M + 1):
            valid = True
            new_diffs = []
            for a in A:
                d = abs(x - a)
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            covers_missing = 0
            for d in new_diffs:
                if d in missing_diffs:
                    covers_missing += 1
            
            candidates.append((x, covers_missing))
            
        if not candidates:
            return None

        # Sort by covers_missing descending
        candidates.sort(key=lambda item: item[1], reverse=True)
        
        # Try all candidates, but limit non-covering ones if too many
        # To prevent explosion, if we have many 0-covering candidates, 
        # maybe only try the first few?
        # But for now, let's try all. M is small.
        
        for x, cov in candidates:
            new_A = A + [x]
            new_A.sort()
            
            new_diffs = set()
            for a in A:
                d = abs(x - a)
                new_diffs.add(d)
            
            new_missing = set(missing_diffs)
            new_missing.difference_update(new_diffs)
            
            new_current = set(current_diffs)
            new_current.update(new_diffs)
            
            result = dfs(new_A, M, new_current, new_missing)
            if result is not None:
                return result
                
        return None

    for M in range(30, 200):
        if time.time() > deadline:
            break
            
        A = [0]
        current_diffs = set()
        missing_diffs = set(range(1, 31))
        
        sol = dfs(A, M, current_diffs, missing_diffs)
        
        if sol is not None:
            if is_valid_solution(M, sol):
                return sol
                
    fallback = [2**i for i in range(19)]
    return fallback
