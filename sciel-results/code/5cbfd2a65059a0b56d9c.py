import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    N = 180
    elements = list(range(1, N + 1))
    
    # Precompute conflicts: for each a, find pairs (b, c) with b > a, c > b
    # such that 1/a = 1/b + 1/c
    # This means b*c = a*(b+c), so bc - ab - ac = 0, (b-a)(c-a) = a^2
    # Let d = b-a, e = c-a, then d*e = a^2, d < e (since b < c)
    # b = a+d, c = a+e, need b,c <= N
    
    # conflicts[a] = set of (b,c) pairs
    # For local search, we need to know: if we have a set S, what triples are violated?
    # A triple (a,b,c) is violated if a,b,c are all in S and 1/a = 1/b+1/c (with distinct elements)
    # Note: a must be the smallest of the three in the equation 1/a = 1/b+1/c
    # Actually, 1/a = 1/b + 1/c implies a < b and a < c (since 1/a > 1/b and 1/a > 1/c)
    # So a is always the smallest element in the triple.
    
    # Precompute all valid triples (a,b,c) with a < b < c and 1/a = 1/b+1/c
    triples = []  # list of (a, b, c)
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N:
                triples.append((a, b, c))
    
    # Build adjacency: for each element x, which triples does it participate in?
    # triple_index[x] = list of indices into triples where x appears
    triple_index = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c) in enumerate(triples):
        triple_index[a].append(i)
        triple_index[b].append(i)
        triple_index[c].append(i)
    
    # Count how many elements are in a set, and count violations
    # For a set S (represented as a boolean array or set), a triple is violated if all 3 are in S.
    
    # Strategy: randomized greedy construction + local search (hill climbing with swaps)
    
    # Greedy construction: build set by adding elements that don't create conflicts
    # We want to maximize size, so we prefer adding elements with fewer conflicts
    
    def greedy_construct():
        S = set()
        # Order elements by some heuristic - try multiple random orderings
        order = elements[:]
        random.shuffle(order)
        
        for x in order:
            # Check if adding x creates any new violation
            # A new violation would be a triple where x is one of the three and the other two are in S
            new_violation = False
            for ti in triple_index[x]:
                a, b, c = triples[ti]
                # Check if the other two elements are in S
                others = [y for y in (a, b, c) if y != x]
                if len(others) == 2 and others[0] in S and others[1] in S:
                    new_violation = True
                    break
            if not new_violation:
                S.add(x)
        return S
    
    def count_violations(S):
        """Count number of violated triples in set S."""
        count = 0
        for a, b, c in triples:
            if a in S and b in S and c in S:
                count += 1
        return count
    
    def local_search(S, max_iterations=50000):
        """Hill climbing: try swaps (remove one, add one) to improve."""
        S = set(S)
        current_violations = count_violations(S)
        best_S = set(S)
        best_violations = current_violations
        best_size = len(S)
        
        # We want to maximize size while keeping violations = 0
        # So the objective is: maximize size subject to violations = 0
        # Or equivalently, minimize violations, then maximize size
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try all possible swaps: remove x from S, add y not in S
            S_list = list(S)
            not_in_S = [x for x in elements if x not in S]
            
            if not not_in_S:
                break
            
            # Randomly sample some swaps for speed
            num_attempts = min(100, len(S_list) * len(not_in_S))
            for _ in range(num_attempts):
                x = random.choice(S_list)
                y = random.choice(not_in_S)
                if x == y:
                    continue
                
                # Try swap: S' = S - {x} + {y}
                # Count change in violations
                # Violations affected: those involving x or y
                # This is complex, so just compute directly for small sets
                S_new = S.copy()
                S_new.discard(x)
                S_new.add(y)
                
                v_new = count_violations(S_new)
                
                # Prefer: fewer violations, then larger size
                if v_new < best_violations or (v_new == best_violations and len(S_new) > best_size):
                    S = S_new
                    best_S = set(S)
                    best_violations = v_new
                    best_size = len(S)
                    improved = True
                    break
                elif v_new == best_violations and len(S_new) > best_size:
                    S = S_new
                    best_S = set(S)
                    best_size = len(S)
                    improved = True
                    break
            
            if not improved:
                # Try single removals to reduce violations if any
                if best_violations > 0:
                    removed = False
                    for x in S_list:
                        S_test = S.copy()
                        S_test.discard(x)
                        v_test = count_violations(S_test)
                        if v_test < best_violations:
                            S = S_test
                            best_S = set(S)
                            best_violations = v_test
                            best_size = len(S)
                            removed = True
                            break
                    if not removed:
                        break
                else:
                    break
        
        return best_S, best_violations, best_size
    
    # Run multiple rounds of greedy + local search
    best_result = set()
    best_score = -1  # score = size if violations == 0, else -infinity
    
    for round_num in range(50):
        if time.time() > deadline:
            break
        
        S = greedy_construct()
        S, violations, size = local_search(S, max_iterations=20000)
        
        if violations == 0:
            if size > best_score:
                best_score = size
                best_result = set(S)
        else:
            # Even with violations, track for potential repair
            pass
    
    # If no valid set found, fall back to a simple valid set
    if not best_result:
        # The original seed: [91, 180] has 90 elements
        # Verify it's valid
        S = set(range(91, 181))
        if count_violations(S) == 0:
            best_result = S
            best_score = len(S)
        else:
            # Find a valid subset
            best_result = set()
    
    return list(best_result)
