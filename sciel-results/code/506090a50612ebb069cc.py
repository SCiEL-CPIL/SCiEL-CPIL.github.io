import time
import random
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Precompute divisibility: a divides (b+c)
    # For each a in 1..n, for each b in 1..n, for each c in 1..n,
    # check if (b+c) % a == 0
    # We'll use this for fast violation checks.
    
    # Precompute for each a, the set of pairs (b,c) with b>c>=a? No, b<=c.
    # Actually, for constraint: a < b <= c, and a | (b+c)
    # Let's precompute for each a, and each b>a, the set of c>=b such that a|(b+c)
    
    # Better: for fast checking, when we have set S, we need to check
    # for all a<b<=c in S: a does NOT divide b+c
    
    # Let's precompute a "conflict" structure.
    # For each pair (b,c) with b<=c, and each a<b, check if a|(b+c)
    # This is O(n^3) = 256^3 = 16.7M, which is feasible once.
    
    # Actually, let's think about this differently for speed.
    
    # Key insight: the upper tail [n/2+1, n] has no violations because
    # for a >= n/2+1, b > a means b >= a+1, c >= b, so b+c >= 2a+2 > a.
    # But a can divide b+c. E.g., a=128, b=129, c=129: b+c=258, 258%128 != 0.
    # But a=85, b=86, c=86: b+c=172, 172%85 != 0.
    # Actually for a > n/2, b > a, c >= b, b+c > 2a > a, and b+c < 2n.
    # a can divide b+c if b+c = 2a, 3a, ... but b+c <= 2n < 2*(2a) since a > n/2.
    # So b+c can equal 2a. When b+c=2a with b>a, c>=b: b>a and c=2a-b<a, contradicting c>=b>a.
    # Wait: b>a, c>=b, so b+c > 2a. And b+c <= 2n. Since a>n/2, 2a>n.
    # b+c > 2a > n, and b+c <= 2n. So b+c can be 2a or 3a (if 3a <= 2n).
    # For a > n/2: 2a > n, 3a > 1.5n. If a > 2n/3, then 3a > 2n, so only possible multiple is 2a.
    # But we showed b+c=2a requires c<a, contradiction. So for a > 2n/3, no violations with a as the small element.
    # For n/2 < a <= 2n/3: 2a and 3a are possible.
    
    # Let's just do a smart search.
    
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
    
    def count_violations(S):
        S_list = sorted(S)
        cnt = 0
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        cnt += 1
        return cnt
    
    def is_valid_add(S, m):
        """Check if adding m to S preserves validity."""
        # m could be a, b, or c in the constraint a<b<=c
        S_list = sorted(S)
        
        # Case 1: m is a (the smallest in the triple)
        # Need: for all b,c in S with b>m, c>=b: m does NOT divide (b+c)
        for b in S_list:
            if b <= m:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % m == 0:
                    return False
        
        # Case 2: m is b (the middle)
        # Need: for all a,c in S with a<m, c>=m: a does NOT divide (m+c)
        # Also c could be m itself (since m is now in the set)
        for a in S_list:
            if a >= m:
                continue
            # c in S with c >= m
            for c in S_list:
                if c < m:
                    continue
                if (m + c) % a == 0:
                    return False
            # c = m
            if (m + m) % a == 0:
                return False
        
        # Case 3: m is c (the largest)
        # Need: for all a,b in S with a<b<=m: a does NOT divide (b+m)
        # Also b could be m itself
        for a in S_list:
            for b in S_list:
                if a >= b:
                    continue
                if b > m:
                    continue
                if (b + m) % a == 0:
                    return False
            # b = m, a < m
            if (m + m) % a == 0:
                return False
        
        return True
    
    def find_violations_fast(S):
        """Find all violating triples, return list of (a,b,c)"""
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
    
    best_S = set()
    best_size = 0
    
    # Strategy: Start from various initial sets, prune violations, then greedily add
    # Also try simulated annealing style local search
    
    # Precompute: for each a, the residues b mod a matter
    # For speed in violation checking, we can use:
    # For fixed a and b, c must satisfy c ≡ -b (mod a) and c >= b
    
    # Let's try multiple initial configurations
    
    # Initial configs: upper tail with different starting points
    for start_val in range(100, 257, 3):
        if time.time() > deadline:
            break
            
        # Start with [start_val, n]
        S = set(range(start_val, n + 1))
        
        # Phase 1: Iteratively remove elements causing most violations
        max_iter = 50
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            V = find_violations_fast(S)
            if not V:
                break
            
            # Count how many times each element appears in violations
            counts = {}
            for a, b, c in V:
                counts[a] = counts.get(a, 0) + 1
                counts[b] = counts.get(b, 0) + 1
                counts[c] = counts.get(c, 0) + 1
            
            # Remove the element with highest count (tie-break: larger value, as it may block more additions)
            max_elem = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
            S.remove(max_elem)
        
        # Phase 2: Greedily add elements from small to large
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for m in range(1, n + 1):
                if m in S:
                    continue
                if is_valid_add(S, m):
                    S.add(m)
                    improved = True
                    break
        
        # Phase 3: Local search - try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = sorted(S)
            for i in range(len(S_list)):
                for j in range(1, n + 1):
                    if j in S:
                        continue
                    # Try swapping S_list[i] with j
                    new_S = S - {S_list[i]} | {j}
                    if is_valid_add(new_S, j):
                        # Also need to verify the removal doesn't create issues (it can't, removing only helps)
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Additional: Try a more aggressive local search from the best found
    if time.time() < deadline - 5:
        # Simulated annealing style: try random removals and re-additions
        current_S = set(best_S)
        current_size = len(current_S)
        
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Randomly try to swap an element
            if current_S:
                remove_elem = random.choice(sorted(current_S))
                add_elem = random.randint(1, n)
                if add_elem in current_S:
                    continue
                
                new_S = current_S - {remove_elem} | {add_elem}
                if is_valid_add(new_S, add_elem):
                    # Check if we gained size (swap keeps same size, but might enable future additions)
                    current_S = new_S
            
            # Try adding any element
            for m in range(1, n + 1):
                if m in current_S:
                    continue
                if is_valid_add(current_S, m):
                    current_S.add(m)
                    break
            
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
    
    return sorted(best_S)
