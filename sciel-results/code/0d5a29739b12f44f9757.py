"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = set()
    for e in P['edges']:
        edges.add(tuple(e))
        edges.add(tuple(reversed(e)))
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85
    
    def is_clique(S):
        S = list(S)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if S[j] not in adj[S[i]]:
                    return False
        return True
    
    def is_independent(S):
        S = list(S)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if S[j] in adj[S[i]]:
                    return False
        return True
    
    def is_valid_part(S):
        return is_clique(S) or is_independent(S)
    
    def solve_k(k):
        """Try to find a partition into exactly k nonempty parts."""
        parts = [[] for _ in range(k)]
        
        def backtrack(v):
            if time.time() > deadline:
                return None
            if v == n:
                # Check all parts are nonempty and valid
                for i in range(k):
                    if not parts[i] or not is_valid_part(parts[i]):
                        return None
                return [list(parts[i]) for i in range(k)]
            
            for pi in range(k):
                trial = parts[pi] + [v]
                if is_valid_part(trial):
                    parts[pi].append(v)
                    res = backtrack(v + 1)
                    if res is not None:
                        return res
                    parts[pi].pop()
            return None
        
        return backtrack(0)
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        result = solve_k(k)
        if result is not None:
            return result
    
    # Fallback
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            if is_valid_part(trial):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
