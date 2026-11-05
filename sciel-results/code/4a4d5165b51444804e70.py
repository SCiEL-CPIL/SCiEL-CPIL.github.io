import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    N = 180
    # Precompute conflicts: a -> list of (b, c) such that {a, b, c} is forbidden
    # a is the smallest element.
    conflicts = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Find divisors of a^2
        # u * v = a^2
        # b = a + u, c = a + v
        # b, c must be > a, distinct, <= N
        # u != v
        for u in range(1, int(a_sq**0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u == v:
                    continue
                b = a + u
                c = a + v
                if b > N or c > N:
                    continue
                if b == c:
                    continue
                # Ensure we don't add duplicate pairs (b,c) and (c,b)
                # Since u != v, b != c.
                # The pair {b,c} is unique for this a and divisor pair {u,v}.
                conflicts[a].append((b, c))
                
    # Helper to check if a set S is valid
    def is_valid(S):
        S_set = set(S)
        for a in S_set:
            for b, c in conflicts[a]:
                if b in S_set and c in S_set:
                    return False
        return True

    # Helper to check if y can be added to S
    def can_add(y, S):
        S_set = set(S)
        # y is smallest
        for b, c in conflicts[y]:
            if b in S_set and c in S_set:
                return False
        # y is larger
        for a in S_set:
            if a >= y:
                continue
            num = a * y
            den = y - a
            if num % den != 0:
                continue
            c = num // den
            if c == a or c == y:
                continue
            if c in S_set:
                return False
        return True

    # Greedy Removal
    S = set(range(1, N + 1))
    
    while time.time() < deadline:
        # Find violations
        violations = []
        counts = {i: 0 for i in S}
        S_set = set(S)
        for a in S_set:
            for b, c in conflicts[a]:
                if b in S_set and c in S_set:
                    violations.append((a, b, c))
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
        
        if not violations:
            break
        
        # Remove element with max count
        max_count = max(counts.values())
        candidates = [x for x in S if counts[x] == max_count]
        # Tie-break: remove smallest? Or random?
        # Smallest numbers are in more triples generally.
        x_to_remove = min(candidates)
        S.remove(x_to_remove)
        
    # Greedy Addition
    # Try to add back elements
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for y in range(1, N + 1):
            if y not in S:
                if can_add(y, S):
                    S.add(y)
                    changed = True
    
    # Local Search / Refinement
    # Try to improve by swapping or kick moves
    best_S = set(S)
    best_size = len(S)
    
    while time.time() < deadline - 5:
        current_S = set(S)
        current_size = len(S)
        
        # Try to add any element
        improved = False
        for y in range(1, N + 1):
            if y not in current_S:
                if can_add(y, current_S):
                    current_S.add(y)
                    improved = True
                    break # Restart loop with new S
        
        if improved:
            S = current_S
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            continue
        
        # If no addition possible, try a kick: remove one, add others
        # Pick a random element to remove
        if not S:
            break
        x = random.choice(list(S))
        S.remove(x)
        
        # Try to add elements
        kick_improved = False
        for y in range(1, N + 1):
            if y not in S:
                if can_add(y, S):
                    S.add(y)
                    kick_improved = True
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
            
        # If we didn't improve size, we might want to revert if it didn't help
        # But we keep S as is for next iteration, as structure might be better.
        # However, if we just removed x and added nothing, we are worse.
        if len(S) < current_size:
             # Revert? Or keep?
             # If we keep, we might get stuck.
             # Let's revert if we didn't gain size.
             S.add(x)
             # Try a different x?
             # To avoid infinite loop of reverting, we can just try another random x next time.
             pass

    return list(best_S)
