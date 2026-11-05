import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    def time_left():
        return deadline - time.time()
    
    def generate_triples(n):
        """Generate all triples in lexicographic order."""
        return list(combinations(range(n), 3))
    
    def triple_index(n):
        """Map triple (a,b,c) to index in lexicographic order."""
        # Precompute or compute on the fly
        pass
    
    def count_violations(colors, n):
        """Count red K4 and blue K5 violations."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check red K4
        red_k4 = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a,b,c)]] == 0 and
                colors[triple_to_idx[(a,b,d)]] == 0 and
                colors[triple_to_idx[(a,c,d)]] == 0 and
                colors[triple_to_idx[(b,c,d)]] == 0):
                red_k4 += 1
        
        # Check blue K5
        blue_k5 = 0
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if colors[triple_to_idx[t]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_k5 += 1
        
        return red_k4 + blue_k5
    
    def local_search(n, max_time=30):
        """Run local search to find valid coloring for n vertices."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute which triples are in which 4-sets and 5-sets
        # For each triple, list of 4-sets it belongs to
        triple_to_quads = {t: [] for t in triples}
        for quad in combinations(range(n), 4):
            for t in combinations(quad, 3):
                triple_to_quads[t].append(quad)
        
        # For each triple, list of 5-sets it belongs to
        triple_to_quints = {t: [] for t in triples}
        for quint in combinations(range(n), 5):
            for t in combinations(quint, 3):
                triple_to_quints[t].append(quint)
        
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Count red triples per 4-set and blue triples per 5-set
        quad_red_count = {}
        for quad in combinations(range(n), 4):
            count = 0
            for t in combinations(quad, 3):
                if colors[triple_to_idx[t]] == 0:
                    count += 1
            quad_red_count[quad] = count
        
        quint_blue_count = {}
        for quint in combinations(range(n), 5):
            count = 0
            for t in combinations(quint, 3):
                if colors[triple_to_idx[t]] == 1:
                    count += 1
            quint_blue_count[quint] = count
        
        def total_violations():
            return sum(1 for v in quad_red_count.values() if v == 4) + \
                   sum(1 for v in quint_blue_count.values() if v == 10)
        
        best_colors = colors[:]
        best_violations = total_violations()
        
        if best_violations == 0:
            return best_colors
        
        search_start = time.time()
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() - search_start < max_time and no_improve < max_no_improve:
            # Find a violating configuration
            violating_triple = None
            for quad in combinations(range(n), 4):
                if quad_red_count[quad] == 4:
                    # Pick a random triple from this quad to flip
                    t = random.choice(list(combinations(quad, 3)))
                    violating_triple = t
                    break
            if violating_triple is None:
                for quint in combinations(range(n), 5):
                    if quint_blue_count[quint] == 10:
                        t = random.choice(list(combinations(quint, 3)))
                        violating_triple = t
                        break
            
            if violating_triple is None:
                break
            
            idx = triple_to_idx[violating_triple]
            old_color = colors[idx]
            new_color = 1 - old_color
            
            # Update counts
            # This triple was red, now blue (or vice versa)
            for quad in triple_to_quads[violating_triple]:
                if old_color == 0:
                    quad_red_count[quad] -= 1
                else:
                    quad_red_count[quad] += 1
            
            for quint in triple_to_quints[violating_triple]:
                if old_color == 1:
                    quint_blue_count[quint] -= 1
                else:
                    quint_blue_count[quint] += 1
            
            colors[idx] = new_color
            
            new_violations = total_violations()
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                no_improve = 0
            else:
                no_improve += 1
                # Revert
                colors[idx] = old_color
                for quad in triple_to_quads[violating_triple]:
                    if old_color == 0:
                        quad_red_count[quad] += 1
                    else:
                        quad_red_count[quad] -= 1
                for quint in triple_to_quints[violating_triple]:
                    if old_color == 1:
                        quint_blue_count[quint] += 1
                    else:
                        quint_blue_count[quint] -= 1
        
        return best_colors
    
    # Try increasing n from a reasonable starting point
    best_n = 9
    best_colors = None
    
    # Start with n=9 (known valid from seed)
    # Then try to increase
    
    # For efficiency, let's try a range of n values
    # The published result is 34, so aim for that
    
    # Try n from 10 up to 40, but with time budget
    for n in range(10, 41):
        if time_left() < 10:
            break
        
        # Try a few random restarts for this n
        found = False
        for attempt in range(3):
            if time_left() < 5:
                break
            
            colors = local_search(n, max_time=min(20, time_left() - 2))
            
            if colors is not None:
                # Verify
                triples = list(combinations(range(n), 3))
                triple_to_idx = {t: i for i, t in enumerate(triples)}
                
                # Check red K4
                valid = True
                for quad in combinations(range(n), 4):
                    a, b, c, d = quad
                    if (colors[triple_to_idx[(a,b,c)]] == 0 and
                        colors[triple_to_idx[(a,b,d)]] == 0 and
                        colors[triple_to_idx[(a,c,d)]] == 0 and
                        colors[triple_to_idx[(b,c,d)]] == 0):
                        valid = False
                        break
                
                if valid:
                    # Check blue K5
                    for quint in combinations(range(n), 5):
                        all_blue = True
                        for t in combinations(quint, 3):
                            if colors[triple_to_idx[t]] != 1:
                                all_blue = False
                                break
                        if all_blue:
                            valid = False
                            break
                
                if valid:
                    best_n = n
                    best_colors = colors
                    found = True
                    break
        
        if not found and n > 10:
            # If we can't find valid for this n, try smaller
            break
    
    if best_colors is None:
        # Fallback to n=9
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
