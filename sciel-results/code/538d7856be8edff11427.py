import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def get_circulant_colors(n, diff_colors):
        """
        diff_colors: list of length n-1? Or normalized?
        Let's use a dict or list indexed by difference 1..n-1.
        For undirected, f(d) == f(n-d).
        We will define f for 1..n-1 explicitly to be safe, 
        but search only over 1..(n-1)//2.
        """
        colors = [0] * (n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                # Normalize d to 1..n-1
                # Since i < j, d is 1..n-1.
                colors[idx] = diff_colors[d]
                idx += 1
        return colors

    def find_circulant_solution(n):
        """
        Try to find a circulant coloring for K_n.
        Returns list of colors for edges, or None.
        """
        if time.time() > deadline - 10:
            return None
            
        # Determine number of independent differences
        # For n, differences are 1..n-1.
        # Symmetry: d and n-d are equivalent.
        # Independent diffs: 1 .. (n-1)//2
        num_indep = (n - 1) // 2
        if n == 2: num_indep = 1 # d=1
        
        # Generate all unique 6-cycles in terms of difference sets
        # A cycle is a sequence of 6 steps summing to 0.
        # We represent a cycle by the frozenset of its absolute differences (normalized 1..(n-1)//2).
        # Note: A cycle might use the same difference multiple times.
        # So we need a multiset or a counter.
        # Let's use a tuple of sorted differences.
        
        cycle_diffs = set()
        # Generate all 6-cycles in K_n
        # This might be slow for n=17 if done naively (742k cycles).
        # But we can optimize:
        # We only care about the difference signature.
        
        # Efficient way:
        # Iterate over all combinations of 6 vertices? No.
        # Iterate over all sequences of 6 differences d1..d6 from 1..n-1 
        # such that sum(d_i) % n == 0 and the path doesn't revisit vertices.
        # This is also complex.
        
        # Given n is small (<=17), we can just generate all 6-cycles in K_n 
        # and extract their difference signatures.
        # 742k cycles * 6 edges = 4.4M operations. This is fast in Python/NumPy.
        
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]: continue
                path = [v0] + list(perm)
                diffs = []
                valid = True
                for k in range(6):
                    u = path[k]
                    v = path[(k+1)%6]
                    d = abs(u - v)
                    # Normalize d
                    if d > n - d:
                        d = n - d
                    diffs.append(d)
                # Sort diffs to create a canonical signature
                sig = tuple(sorted(diffs))
                cycle_diffs.add(sig)
                
        # cycle_diffs is a set of tuples of length 6.
        
        # Now search for diff_colors for 1..num_indep
        # diff_colors[d] is the color for difference d.
        # For d > num_indep, color is diff_colors[n-d].
        
        # We iterate over all 4^num_indep possibilities.
        # For n=17, num_indep=8. 4^8 = 65536.
        # For each, check if any sig is "monochromatic".
        # A sig is monochromatic if all d in sig have the same color.
        
        for combo_colors in itertools.product([0, 1, 2, 3], repeat=num_indep):
            # combo_colors[d-1] is the color for difference d
            # Map diff to color
            def get_color(d):
                # d is 1..n-1
                if d > num_indep:
                    d = n - d
                return combo_colors[d-1]
            
            is_valid = True
            for sig in cycle_diffs:
                # Check if all diffs in sig have the same color
                c0 = get_color(sig[0])
                for d in sig[1:]:
                    if get_color(d) != c0:
                        break
                else:
                    # All same
                    is_valid = False
                    break
                    
            if is_valid:
                # Construct full diff_colors array
                full_diff_colors = [0] * n # index 0 unused
                for d in range(1, n):
                    if d <= num_indep:
                        full_diff_colors[d] = combo_colors[d-1]
                    else:
                        full_diff_colors[d] = combo_colors[n-d-1]
                        
                return get_circulant_colors(n, full_diff_colors)
                
        return None

    # Try n from 17 down to 13
    for n in [17, 16, 15, 14, 13]:
        if time.time() > deadline - 20:
            break
        colors = find_circulant_solution(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback: General random search for n=13
    # ... (keep the old random search code here as fallback) ...
    
    return {'n': 1, 'colors': []}
