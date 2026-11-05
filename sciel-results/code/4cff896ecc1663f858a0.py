import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85
    
    def check_valid(S):
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def find_violations(S):
        S_list = sorted(S)
        violations = []
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        violations.append((a, b, c))
        return violations

    def is_valid_add(S, m):
        # Check if adding m to S creates a violation
        # m can be a, b, or c
        S_list = sorted(S)
        
        # Case 1: m is a (smallest)
        # Need m ∤ (b+c) for b,c in S, b > m, c >= b
        for b in S_list:
            if b <= m: continue
            for c in S_list:
                if c < b: continue
                if (b+c) % m == 0:
                    return False
        
        # Case 2: m is b (middle)
        # Need a ∤ (m+c) for a in S, a < m, c in S, c >= m
        for a in S_list:
            if a >= m: continue
            for c in S_list:
                if c < m: continue
                if (m+c) % a == 0:
                    return False
            # Also c could be m? No, c >= m. If c=m, (m+m)%a?
            # The loop covers c in S. If m is added, m is not in S yet.
            # So we check c in S.
            # What if c=m? m is not in S.
            # So we only check c in S.
            # But wait, if we add m, m becomes part of S.
            # The condition is for a < b <= c in the NEW set.
            # If b=m, c can be m? Yes, if m is in S.
            # So we must check c=m as well.
            # In the loop above, c iterates over S. m is not in S.
            # So we miss the case c=m.
            # We need to check if (m+m) % a == 0.
            if (2*m) % a == 0:
                 return False

        # Case 3: m is c (largest)
        # Need a ∤ (b+m) for a in S, a < b, b in S, b <= m
        for a in S_list:
            for b in S_list:
                if a >= b: continue
                if b > m: continue
                if (b+m) % a == 0:
                    return False
        # Also if b=m?
        # If b=m, then a < m.
        # The loop iterates b in S. m is not in S.
        # So we miss b=m.
        # We need to check if (m+m) % a == 0 for a < m.
        # This is the same check as in Case 2 for (m+m)%a.
        # So it's covered if we added it in Case 2.

        return True

    best_S = set()
    best_size = 0
    
    # Try different starting sets
    for start_val in range(129, 257, 5):
        if time.time() > deadline:
            break
            
        S = set(range(start_val, n + 1))
        
        # Phase 1: Remove elements to fix violations
        while time.time() < deadline:
            V = find_violations(S)
            if not V:
                break
            
            # Count occurrences
            counts = {}
            for a, b, c in V:
                counts[a] = counts.get(a, 0) + 1
                counts[b] = counts.get(b, 0) + 1
                counts[c] = counts.get(c, 0) + 1
            
            # Remove max
            max_elem = max(counts, key=counts.get)
            S.remove(max_elem)
            
        # Phase 2: Add elements back
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for m in range(1, n + 1):
                if m in S:
                    continue
                if is_valid_add(S, m):
                    S.add(m)
                    improved = True
                    break # Restart scan or continue?
                    # Breaking and restarting ensures we don't miss interactions
                    # But scanning 1..256 is fast.
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    return sorted(best_S)
