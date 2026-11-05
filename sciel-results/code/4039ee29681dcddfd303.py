import numpy as np
import time
from itertools import combinations
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(np.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True

    def build_structure(n):
        """Build indices for K4s and K5s."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        triple_to_k4s = defaultdict(list)
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            k4_idx = len(k4s)
            k4s.append((t1, t2, t3, t4))
            for t in (t1, t2, t3, t4):
                triple_to_k4s[t].append(k4_idx)
        
        k5s = []
        triple_to_k5s = defaultdict(list)
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5_idx = len(k5s)
            k5s.append(tuple(idxs))
            for t in idxs:
                triple_to_k5s[t].append(k5_idx)
                
        return triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s

    def count_violations(colors, k4s, k5s):
        """Count red K4s and blue K5s."""
        red_k4 = 0
        for idxs in k4s:
            # Check if all 4 are 0
            if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                red_k4 += 1
        blue_k5 = 0
        for idxs in k5s:
            # Check if all 10 are 1
            is_blue = True
            for idx in idxs:
                if colors[idx] != 1:
                    is_blue = False
                    break
            if is_blue:
                blue_k5 += 1
        return red_k4 + blue_k5

    def generate_qr_coloring(n, p):
        """
        Generate coloring based on quadratic residues mod p.
        Vertices 0..n-1.
        Color (a,b,c) red (0) if (b-a) and (c-b) are QR mod p, else blue (1).
        """
        qr = set()
        for i in range(1, p):
            qr.add((i * i) % p)
        
        triples = list(combinations(range(n), 3))
        colors = []
        for (a, b, c) in triples:
            d1 = (b - a) % p
            d2 = (c - b) % p
            if d1 in qr and d2 in qr:
                colors.append(0)
            else:
                colors.append(1)
        return np.array(colors, dtype=np.int8)

    def generate_linear_coloring(n, coeffs):
        """
        Color (a,b,c) red (0) if sum(coeffs[i] * v_i) % 2 == 0, else blue (1).
        coeffs is a tuple of length n.
        """
        triples = list(combinations(range(n), 3))
        colors = []
        for (a, b, c) in triples:
            val = (coeffs[a] + coeffs[b] + coeffs[c]) % 2
            colors.append(val)
        return np.array(colors, dtype=np.int8)

    def local_search(colors, n, time_limit, k4s, k5s, triple_to_k4s, triple_to_k5s):
        """
        Local search to minimize violations.
        """
        num_triples = len(colors)
        current_colors = colors.copy()
        current_cost = count_violations(current_colors, k4s, k5s)
        
        if current_cost == 0:
            return current_colors
        
        temp = 1.0
        cool_rate = 0.99
        min_temp = 0.01
        
        while time.time() < time_limit:
            # Pick a random edge
            t = np.random.randint(num_triples)
            old_color = current_colors[t]
            new_color = 1 - old_color
            
            # Calculate delta cost efficiently?
            # For now, just flip and recalc if cost is small, or use delta.
            # Delta calculation is complex for K5.
            # Given n <= 34, num_triples <= 5984.
            # Recalculating full cost is O(num_k4 + num_k5) ~ 300k ops.
            # This is too slow for many iterations.
            
            # We need a faster delta calculation or a different strategy.
            # Let's use a "flip and check" strategy with early exit.
            
            # Actually, let's just do a few flips and recalc.
            # If we can't do fast delta, we can limit iterations.
            
            # Alternative: Greedy repair.
            # Find an edge involved in a violation, flip it.
            
            # Let's stick to SA but optimize the cost check.
            # We can precompute for each edge which K4s and K5s it belongs to.
            # And maintain counts of "almost all red" K4s and "almost all blue" K5s.
            
            # This is getting complicated. Let's try a simpler "Hill Climbing" first.
            # If we get stuck, we use SA.
            
            # For this implementation, we will use a simplified SA that recalculates cost
            # but only accepts if cost decreases, or with low probability if it increases.
            # To make it fast, we will limit the number of steps.
            
            # Let's try to calculate delta properly.
            # Delta for flipping t from c to 1-c:
            # 1. K4s containing t:
            #    If K4 was all red (4 reds), and we flip t to blue, red_count drops by 1. Violation removed (-1).
            #    If K4 had 3 reds and t was blue (not a violation), and we flip t to red, red_count becomes 4. Violation added (+1).
            # 2. K5s containing t:
            #    If K5 was all blue (10 blues), and we flip t to red, blue_count drops by 1. Violation removed (-1).
            #    If K5 had 9 blues and t was red (not a violation), and we flip t to blue, blue_count becomes 10. Violation added (+1).
            
            delta = 0
            # Check K4s
            for k4_idx in triple_to_k4s[t]:
                idxs = k4s[k4_idx]
                # Count reds in this K4 excluding t
                reds_ex_t = 0
                for idx in idxs:
                    if idx != t and current_colors[idx] == 0:
                        reds_ex_t += 1
                
                if new_color == 0:
                    # t becomes red
                    if reds_ex_t == 3:
                        # Becomes all red
                        delta += 1
                else:
                    # t becomes blue
                    if reds_ex_t == 3 and current_colors[t] == 0:
                        # Was all red, now not
                        delta -= 1
            
            # Check K5s
            for k5_idx in triple_to_k5s[t]:
                idxs = k5s[k5_idx]
                # Count blues in this K5 excluding t
                blues_ex_t = 0
                for idx in idxs:
                    if idx != t and current_colors[idx] == 1:
                        blues_ex_t += 1
                
                if new_color == 1:
                    # t becomes blue
                    if blues_ex_t == 9:
                        # Becomes all blue
                        delta += 1
                else:
                    # t becomes red
                    if blues_ex_t == 9 and current_colors[t] == 1:
                        # Was all blue, now not
                        delta -= 1
            
            new_cost = current_cost + delta
            
            if new_cost < current_cost or (new_cost == current_cost and np.random.rand() < 0.5):
                accept = True
            elif temp > min_temp:
                accept = np.random.rand() < np.exp(-new_cost / max(temp, 1e-10))
            else:
                accept = False
            
            if accept:
                current_colors[t] = new_color
                current_cost = new_cost
                if current_cost == 0:
                    return current_colors
            
            temp *= cool_rate
            
            # Occasionally do a greedy repair
            if np.random.rand() < 0.1:
                # Find an edge in a violation and flip it
                # This is heuristic and might be slow, so skip for now or do simple version
                pass

        return current_colors

    # Main logic
    best_n = 9
    best_colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(9),3)]
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 10:
            break
            
        # Build structure
        try:
            triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        except Exception:
            continue
            
        # Generate candidates
        candidates = []
        
        # 1. QR Coloring
        # Try primes near n
        for p in [n, n+1, n+2, n+3, n+4, n+5, 31, 37, 41, 43]:
            if p > n and is_prime(p):
                try:
                    c = generate_qr_coloring(n, p)
                    candidates.append(c)
                except Exception:
                    pass
        
        # 2. Linear Colorings
        # Try a few random coefficient sets
        for _ in range(5):
            coeffs = np.random.randint(0, 2, n)
            c = generate_linear_coloring(n, coeffs)
            candidates.append(c)
            
        # 3. Random
        for _ in range(5):
            c = np.random.randint(0, 2, num_triples, dtype=np.int8)
            candidates.append(c)
            
        # 4. All 0, All 1 (unlikely to work but cheap)
        candidates.append(np.zeros(num_triples, dtype=np.int8))
        candidates.append(np.ones(num_triples, dtype=np.int8))
        
        # Evaluate candidates
        best_candidate = None
        best_cost = float('inf')
        
        for c in candidates:
            cost = count_violations(c, k4s, k5s)
            if cost < best_cost:
                best_cost = cost
                best_candidate = c
            if cost == 0:
                best_n = n
                best_colors = c.tolist()
                break
        
        if best_n == n:
            break
            
        # If we didn't find a perfect one, try local search on the best candidate
        if best_candidate is not None and time.time() < deadline - 20:
            # Allocate time for local search
            ls_time = min(20.0, deadline - time.time() - 10)
            if ls_time > 1:
                try:
                    improved = local_search(best_candidate, n, time.time() + ls_time, k4s, k5s, triple_to_k4s, triple_to_k5s)
                    cost = count_violations(improved, k4s, k5s)
                    if cost == 0:
                        best_n = n
                        best_colors = improved.tolist()
                        break
                    elif cost < best_cost:
                        best_cost = cost
                        best_candidate = improved
                except Exception:
                    pass

    return {'n': best_n, 'colors': best_colors}
