import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        # For g=9, d=4 gives girth 9, n=2*5^4=1250
        # For g=10, d=5 gives girth 10, n=2*5^5=6250
        # We want min n, so use smallest d such that d+5 >= g
        d = max(3, g - 5)
        if d % 2 != (g - 5) % 2: # Adjust parity if needed?
            # Seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
            # For g=9: g-5=4 (even), so d = 9-4=5? 
            # Wait, seed said d=5 for g=9.
            # Let's stick to seed logic to be safe, or use d = g-5.
            # Girth of LUW(d,q) is d+5.
            # If we want girth >= 9, d+5 >= 9 => d >= 4.
            # Seed used d=5. Why?
            # Maybe d must be odd? Or even?
            # In LUW, d is usually >= 3.
            # Let's try d=4.
            pass
        
        # Use seed's logic for d
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Specialized for k=5, g=9
    if k == 5 and g == 9:
        deadline = time.time() + 110 # 10s margin
        
        # Try Cayley graph construction on Z_m
        # S = {a, -a, b, -b, m/2}
        # m must be even.
        
        def check_girth_cyclic(m, a, b):
            # Check if there is a non-backtracking closed walk of length < g
            # Walk: s1, ..., sL in S, sum si = 0, si != -s(i-1)
            # We can use BFS from 0 to find shortest cycle.
            # But we just need to know if min cycle < g.
            
            # Since k=5 is small, we can just generate all non-backtracking walks of length up to g-1
            # and check if any sum to 0.
            
            # To avoid duplicates, we can just check all sequences?
            # Number of sequences: 5 * 4^(L-1).
            # L=8: 5 * 4^7 = 5 * 16384 = 81920.
            # This is fast enough for a single check.
            
            S = [a, m-a, b, m-b, m//2]
            
            # Optimization: 
            # If we find a cycle of length L, we return False.
            # We check L from 3 to g-1.
            
            # We can do a DFS.
            # State: (current_sum, last_step)
            # We want to reach sum=0 with length < g.
            
            # Actually, simpler:
            # Iterate over all non-backtracking walks of length L.
            # Sum them. If any is 0, return False.
            
            # We can prune if the remaining steps cannot possibly sum to 0?
            # Not easy with modular arithmetic.
            
            # Let's just iterate.
            for L in range(3, g):
                # Generate walks
                # We fix the first step to avoid rotation? No, sum is absolute.
                # But we can fix first step to a to reduce work?
                # No, the cycle can start anywhere. But the sum is the same.
                # However, the non-backtracking constraint is relative.
                # Any cycle is a sequence s1...sL.
                # We can just generate all sequences of length L from S with si != -s(i-1).
                
                # To speed up, we can use recursion.
                def dfs(length, current_sum, last_step):
                    if length == L:
                        return current_sum == 0
                    for s in S:
                        if s == -last_step:
                            continue
                        if dfs(length + 1, (current_sum + s) % m, s):
                            return True
                    return False
                
                # We can optimize by checking if it's possible to reach 0.
                # But for now, simple DFS.
                # We can skip the first step symmetry?
                # The set of sums for walks starting with a is the same as starting with -a (shifted?).
                # No.
                
                # Let's just run it.
                # 81920 calls * 8 lengths * m_candidates.
                # If m_candidates is 100, that's 65M calls. Might be too slow in Python.
                
                # Optimization:
                # Precompute all non-backtracking walks of length L as lists of (coeff_a, coeff_b, coeff_half).
                # Sum = ca*a + cb*b + ch*(m/2)
                # We need ca*a + cb*b + ch*(m/2) == 0 mod m.
                
                # This is a linear Diophantine equation mod m.
                
                # Let's generate the coefficient vectors.
                # There are only 5 * 4^(L-1) vectors.
                # We can precompute these vectors for L=3..8 once.
                
                pass # Placeholder
            
            # Better approach:
            # Precompute coefficient vectors for all L < g.
            # For each m, check if any vector (ca, cb, ch) satisfies ca*a + cb*b + ch*(m/2) == 0 mod m.
            
            return True # Placeholder

        # Precompute coefficient vectors
        # S = {a, -a, b, -b, h} where h = m/2.
        # Note: h is its own inverse.
        
        # We represent a walk by (ca, cb, ch).
        # Start with empty walk (0,0,0).
        # Next step s:
        # if s == a: (ca+1, cb, ch)
        # if s == -a: (ca-1, cb, ch)
        # if s == b: (ca, cb+1, ch)
        # if s == -b: (ca, cb-1, ch)
        # if s == h: (ca, cb, ch+1)
        
        # Non-backtracking: s != -last.
        # -a is opposite to a.
        # -b is opposite to b.
        # h is opposite to h.
        
        # So if last was a, next cannot be -a.
        # If last was -a, next cannot be a.
        # If last was b, next cannot be -b.
        # If last was -b, next cannot be b.
        # If last was h, next cannot be h.
        
        coeff_vectors = {L: [] for L in range(3, g)}
        
        # We will generate all non-backtracking walks of length L.
        # To avoid explosion, we note that the number is small.
        
        for L in range(3, g):
            vectors = []
            # We can just use a set to deduplicate vectors?
            # Many walks might have the same coefficient vector.
            # We only need to check if ANY vector satisfies the equation.
            # So we can just collect unique vectors.
            
            unique_vectors = set()
            
            # DFS
            def gen_walks(length, ca, cb, ch, last):
                if length == L:
                    unique_vectors.add((ca, cb, ch))
                    return
                # Next steps
                # Opposite of last
                if last == 'a':
                    # Cannot take -a
                    # Can take a, b, -b, h
                    gen_walks(length+1, ca+1, cb, ch, 'a')
                    gen_walks(length+1, ca, cb+1, ch, 'b')
                    gen_walks(length+1, ca, cb-1, ch, '-b')
                    gen_walks(length+1, ca, cb, ch+1, 'h')
                elif last == '-a':
                    # Cannot take a
                    gen_walks(length+1, ca-1, cb, ch, '-a')
                    gen_walks(length+1, ca, cb+1, ch, 'b')
                    gen_walks(length+1, ca, cb-1, ch, '-b')
                    gen_walks(length+1, ca, cb, ch+1, 'h')
                elif last == 'b':
                    # Cannot take -b
                    gen_walks(length+1, ca+1, cb, ch, 'a')
                    gen_walks(length+1, ca-1, cb, ch, '-a')
                    gen_walks(length+1, ca, cb+1, ch, 'b')
                    gen_walks(length+1, ca, cb, ch+1, 'h')
                elif last == '-b':
                    # Cannot take b
                    gen_walks(length+1, ca+1, cb, ch, 'a')
                    gen_walks(length+1, ca-1, cb, ch, '-a')
                    gen_walks(length+1, ca, cb-1, ch, '-b')
                    gen_walks(length+1, ca, cb, ch+1, 'h')
                elif last == 'h':
                    # Cannot take h
                    gen_walks(length+1, ca+1, cb, ch, 'a')
                    gen_walks(length+1, ca-1, cb, ch, '-a')
                    gen_walks(length+1, ca, cb+1, ch, 'b')
                    gen_walks(length+1, ca, cb-1, ch, '-b')
                else: # start
                    # Can take any
                    gen_walks(length+1, ca+1, cb, ch, 'a')
                    gen_walks(length+1, ca-1, cb, ch, '-a')
                    gen_walks(length+1, ca, cb+1, ch, 'b')
                    gen_walks(length+1, ca, cb-1, ch, '-b')
                    gen_walks(length+1, ca, cb, ch+1, 'h')

            gen_walks(1, 0, 0, 0, None)
            coeff_vectors[L] = list(unique_vectors)
        
        # Search for m, a, b
        best_n = 1250
        best_edges = None
        
        # Moore bound is 425. Try m from 426 to 1000.
        # m must be even.
        for m in range(428, 1000, 2):
            if time.time() > deadline:
                break
            
            h = m // 2
            
            # a, b in [1, m/2 - 1]
            # a != b
            # To reduce work, we can try small a, b first?
            # Or random?
            # Let's try a=1, b=2?
            # Usually "spread out" generators are better.
            # Let's try a=1, b=m/3?
            
            # We will iterate a, b.
            # To make it faster, we can check necessary conditions.
            # For each vector (ca, cb, ch) in coeff_vectors[L],
            # we need ca*a + cb*b + ch*h != 0 mod m.
            
            # We can pre-filter (a,b) pairs?
            
            found = False
            # We can iterate a from 1 to m/2-1
            # and b from a+1 to m/2-1
            
            # Optimization:
            # If we find a valid (a,b), we build the graph and return.
            
            # To speed up, we can randomize the order of (a,b) or use heuristics.
            # Let's try a few specific promising pairs first, then scan.
            
            candidates = []
            # Heuristic: a and b should be "large" relative to m?
            # Or small?
            # In Z_m, small steps create short cycles.
            # So we want a, b to be "large" in some sense?
            # Actually, we want them to be "independent".
            
            # Let's just scan, but break early if we find one.
            # To avoid O(m^2) being too slow, we can limit the number of checks per m.
            # Or use random sampling.
            
            # Let's try random sampling of (a,b) for each m.
            for _ in range(100):
                a = random.randint(1, h-1)
                b = random.randint(1, h-1)
                if a == b: continue
                if a == h: continue # a must be < h to be distinct from -a? 
                # In Z_m, a and -a are distinct unless 2a=0 => a=m/2.
                # So a in [1, m/2-1] is fine.
                
                valid = True
                for L in range(3, g):
                    for (ca, cb, ch) in coeff_vectors[L]:
                        val = (ca * a + cb * b + ch * h) % m
                        if val == 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Found a valid (m, a, b)
                    # Build the graph
                    n = m
                    edges = []
                    S = [a, m-a, b, m-b, h]
                    
                    # Generate edges
                    # For each u in 0..m-1, for each s in S:
                    # Edge (u, (u+s)%m)
                    # Add if u < (u+s)%m to avoid duplicates?
                    # Since S is symmetric, (u, u+s) and (u+s, u) are the same.
                    # We can just add all (u, (u+s)%m) and then deduplicate?
                    # Or just add if u < v.
                    
                    for u in range(m):
                        for s in S:
                            v = (u + s) % m
                            if u < v:
                                edges.append((u, v))
                    
                    # Verify degree
                    # Should be 5.
                    
                    return n, edges
        
        # If no solution found, return fallback
        return luw_construct(k, g)
        
    else:
        return luw_construct(k, g)

# EVOLVE-BLOCK-END
