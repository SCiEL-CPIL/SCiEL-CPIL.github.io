import time
import random

def construct():
    t0 = time.time()
    deadline = 55.0
    N = 128
    
    def elapsed():
        return time.time() - t0
    
    # Phase 1: Greedy removal
    S = set(range(1, N + 1))
    in_S = [False] * (N + 1)
    for x in S:
        in_S[x] = True
        
    def has_violation(S, in_S):
        """Check if S contains any violating triple. Returns the triple if found, else None."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                ab = a + b
                if ab > N:
                    break
                if not in_S[ab]:
                    continue
                for k in range(j + 1, n):
                    c = S_list[k]
                    ac = a + c
                    if not in_S[ac]:
                        continue
                    bc = b + c
                    if in_S[bc]:
                        return (a, b, c)
        return None

    def count_violations_for_elem(S, in_S, x):
        """Count how many violating triples involve x."""
        count = 0
        S_list = [e for e in S if e != x]
        n = len(S_list)
        
        # Case 1: x is one of a, b, c
        # We need u, v in S\{x} such that x+u in S, x+v in S, u+v in S
        # Note: u+v must be in S. x+u, x+v in S.
        for i in range(n):
            u = S_list[i]
            xu = x + u
            if xu > N or not in_S[xu]:
                continue
            for j in range(i + 1, n):
                v = S_list[j]
                xv = x + v
                if xv > N or not in_S[xv]:
                    continue
                uv = u + v
                if in_S[uv]:
                    count += 1
        
        # Case 2: x is one of the sums
        # x = a + b. Need c in S\{a,b} such that a+c in S and b+c in S.
        # a, b, c distinct.
        for a in S_list:
            b = x - a
            if b <= 0 or b > N:
                continue
            if b == a or b == x:
                continue
            if not in_S[b]:
                continue
            # Found pair (a,b) summing to x. Check for c.
            for c in S_list:
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                if ac <= N and bc <= N and in_S[ac] and in_S[bc]:
                    count += 1
                    break # One c is enough to count this (a,b) pair? 
                    # Wait, the definition is "forbids THREE DISTINCT members a,b,c for which...".
                    # If there are multiple c's, they form multiple triples.
                    # So we should count all valid c's.
                    # But for heuristic, finding one is enough to say "bad".
                    # For counting, we need all.
        
        return count

    # Greedy removal loop
    while elapsed() < deadline - 15.0:
        # Find violation
        viol = has_violation(S, in_S)
        if not viol:
            break
        
        # If we want to be smarter, we could count violations for each element in the triple
        # and remove the one with highest count.
        # For now, let's stick to removing one of them, maybe the one that appears in most violations?
        # To keep it fast, let's just remove the first one or randomize.
        # Actually, let's compute count for the 3 elements in the violation.
        a, b, c = viol
        candidates = [a, b, c]
        best_elem = None
        best_count = -1
        for x in candidates:
            cnt = count_violations_for_elem(S, in_S, x)
            if cnt > best_count:
                best_count = cnt
                best_elem = x
        
        S.remove(best_elem)
        in_S[best_elem] = False

    # Phase 2: Add back
    removed = [x for x in range(1, N + 1) if not in_S[x]]
    random.shuffle(removed)
    
    def can_add(S, in_S, x):
        """Check if adding x to S keeps it valid."""
        if in_S[x]:
            return False
            
        # Case 1: x is one of a, b, c
        # Need u, v in S such that x+u in S, x+v in S, u+v in S.
        # u, v distinct, u!=x, v!=x.
        # Since x is not in S yet, u,v are just in S.
        S_list = list(S)
        # Optimization: Precompute U_x = {u in S | x+u in S}
        U_x = []
        for u in S_list:
            xu = x + u
            if xu <= N and in_S[xu]:
                U_x.append(u)
                
        # Check pairs in U_x
        for i in range(len(U_x)):
            u = U_x[i]
            for j in range(i + 1, len(U_x)):
                v = U_x[j]
                uv = u + v
                if in_S[uv]:
                    return False

        # Case 2: x is one of the sums
        # x = a + b. Need c in S such that a+c in S and b+c in S.
        # a, b, c distinct. a,b in S.
        for a in S_list:
            b = x - a
            if b <= 0 or b > N or b == a:
                continue
            if not in_S[b]:
                continue
            # Check for c
            for c in S_list:
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                if ac <= N and bc <= N and in_S[ac] and in_S[bc]:
                    return False

        return True

    for x in removed:
        if elapsed() > deadline - 10.0:
            break
        if can_add(S, in_S, x):
            S.add(x)
            in_S[x] = True

    # Phase 3: Local search
    best_size = len(S)
    no_improve = 0
    
    while elapsed() < deadline - 2.0:
        improved = False
        
        # Try adding
        for x in range(1, N + 1):
            if in_S[x] or elapsed() > deadline - 2.0:
                continue
            if can_add(S, in_S, x):
                S.add(x)
                in_S[x] = True
                improved = True
                break
        
        if improved:
            best_size = len(S)
            no_improve = 0
            continue
        
        # Try swaps
        S_list = list(S)
        random.shuffle(S_list)
        for y in S_list:
            if elapsed() > deadline - 2.0:
                break
            S.remove(y)
            in_S[y] = False
            
            found = False
            for x in range(1, N + 1):
                if x == y or in_S[x]:
                    continue
                if can_add(S, in_S, x):
                    S.add(x)
                    in_S[x] = True
                    found = True
                    improved = True
                    break
            
            if not found:
                S.add(y)
                in_S[y] = True
        
        if improved:
            best_size = len(S)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                break

    result = sorted([int(x) for x in S])
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
