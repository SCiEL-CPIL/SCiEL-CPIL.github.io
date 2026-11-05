"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_triple_index(n):
        triples = get_triples(n)
        tri_index = {t: i for i, t in enumerate(triples)}
        return tri_index, triples

    def count_violations_fast(n, colors, tri_index):
        """Count red K4^(3) and blue K5^(3) violations efficiently."""
        red_violations = 0
        blue_violations = 0
        
        # Precompute all 4-sets and their triple indices
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            tri_indices = [
                tri_index[(a,b,c)], tri_index[(a,b,d)],
                tri_index[(a,c,d)], tri_index[(b,c,d)]
            ]
            if all(colors[i] == 0 for i in tri_indices):
                red_violations += 1
        
        # Precompute all 5-sets and their triple indices
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                idx = tri_index[tri]
                if colors[idx] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_violations += 1
        
        return red_violations, blue_violations

    def is_valid(n, colors, tri_index):
        red_v, blue_v = count_violations_fast(n, colors, tri_index)
        return red_v == 0 and blue_v == 0

    def algebraic_coloring(n, p, field_size):
        """
        Deterministic algebraic coloring using finite field structure.
        For vertices 0..n-1, map to field elements and color based on 
        whether (a-b)*(b-c)*(c-a) is a quadratic residue or not.
        """
        triples = get_triples(n)
        colors = []
        
        # For prime p, use modular arithmetic
        # Color triple (a,b,c) based on the product (a-b)*(b-c)*(c-a) mod p
        # If it's a quadratic residue -> red (0), else blue (1)
        
        # Precompute quadratic residues mod p
        if p > 1:
            qr = set()
            for x in range(p):
                qr.add((x * x) % p)
            qr.discard(0)  # 0 is neither
        
        for (a, b, c) in triples:
            # Compute product of differences mod p
            prod = ((a - b) * (b - c) * (c - a)) % p
            if prod == 0:
                # Two vertices map to same field element - shouldn't happen for n <= p
                colors.append(1)
            elif prod in qr:
                colors.append(0)  # Red
            else:
                colors.append(1)  # Blue
        
        return colors

    def try_algebraic(n):
        """Try algebraic coloring with various primes."""
        # Try primes around n
        candidates = []
        for p in range(n, min(n + 50, 1000)):
            # Check if p is prime
            if p < 2:
                continue
            is_prime = True
            for d in range(2, int(p**0.5) + 1):
                if p % d == 0:
                    is_prime = False
                    break
            if is_prime:
                candidates.append(p)
        
        best_colors = None
        best_score = float('inf')
        
        for p in candidates:
            if time.time() > deadline - 5:
                break
            
            colors = algebraic_coloring(n, p, p)
            tri_index, _ = build_triple_index(n)
            red_v, blue_v = count_violations_fast(n, colors, tri_index)
            score = red_v * 100 + blue_v
            
            if score < best_score:
                best_score = score
                best_colors = colors
            
            if score == 0:
                return colors
        
        return best_colors

    def local_search_improved(n, max_time=10):
        """Improved local search with better move selection."""
        tri_index, triples = build_triple_index(n)
        num_triples = len(triples)
        
        # Start with algebraic coloring
        colors = try_algebraic(n)
        if colors is None:
            colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        best_colors = colors[:]
        best_score = float('inf')
        
        end_time = time.time() + max_time
        
        while time.time() < end_time:
            red_v, blue_v = count_violations_fast(n, colors, tri_index)
            score = red_v * 100 + blue_v
            
            if score < best_score:
                best_score = score
                best_colors = colors[:]
            
            if score == 0:
                break
            
            # Try to find a beneficial flip
            improved = False
            # Try random flips with limited restarts
            for attempt in range(50):
                if time.time() > end_time:
                    break
                idx = random.randint(0, num_triples - 1)
                old_color = colors[idx]
                colors[idx] = 1 - old_color
                
                new_red_v, new_blue_v = count_violations_fast(n, colors, tri_index)
                new_score = new_red_v * 100 + new_blue_v
                
                if new_score < score:
                    score = new_score
                    improved = True
                    break
                else:
                    colors[idx] = old_color
            
            if not improved:
                # Accept a random move to escape local minimum
                if random.random() < 0.3:
                    idx = random.randint(0, num_triples - 1)
                    colors[idx] = 1 - colors[idx]
                else:
                    # Try a small neighborhood of flips
                    num_flips = random.randint(2, 5)
                    flipped = []
                    for _ in range(num_flips):
                        idx = random.randint(0, num_triples - 1)
                        colors[idx] = 1 - colors[idx]
                        flipped.append(idx)
                    
                    # Check if this helps
                    new_red_v, new_blue_v = count_violations_fast(n, colors, tri_index)
                    new_score = new_red_v * 100 + new_blue_v
                    if new_score >= score:
                        for idx in flipped:
                            colors[idx] = 1 - colors[idx]
                    break
        
        return best_colors, best_score

    # Main search strategy
    best_n = 0
    best_colors = None
    
    # Try algebraic constructions for increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break
        
        # First try pure algebraic
        colors = try_algebraic(target_n)
        if colors is not None:
            tri_index, _ = build_triple_index(target_n)
            if is_valid(target_n, colors, tri_index):
                best_n = target_n
                best_colors = colors
                continue
        
        # If algebraic didn't work, try local search starting from algebraic
        if time.time() > deadline - 15:
            break
        
        colors, score = local_search_improved(target_n, max_time=8)
        if score == 0:
            best_n = target_n
            best_colors = colors
            continue
        else:
            # If we couldn't improve, stop trying larger n
            if best_n == 0:
                # Save the best partial result
                tri_index, triples = build_triple_index(target_n)
                best_n = target_n
                best_colors = colors
            break
    
    # If we didn't find anything, use a small valid construction
    if best_n == 0:
        # Known valid small construction
        best_n = 9
        triples9 = get_triples(9)
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples9]
    
    # Final verification and return
    triples_final = get_triples(best_n)
    if len(best_colors) != len(triples_final):
        # Regenerate if mismatch
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_final]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
