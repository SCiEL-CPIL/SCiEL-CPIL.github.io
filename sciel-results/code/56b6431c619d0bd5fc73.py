import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    def precompute_sets(n):
        triples = get_triples(n)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        triple_to_4sets = [[] for _ in range(len(triples))]
        triple_to_5sets = [[] for _ in range(len(triples))]
        
        for i, s4 in enumerate(four_sets):
            a, b, c, d = s4
            t1 = (a, b, c)
            t2 = (a, b, d)
            t3 = (a, c, d)
            t4 = (b, c, d)
            triple_to_4sets[triple_index[t1]].append(i)
            triple_to_4sets[triple_index[t2]].append(i)
            triple_to_4sets[triple_index[t3]].append(i)
            triple_to_4sets[triple_index[t4]].append(i)
            
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                triple_to_5sets[triple_index[t]].append(i)
                
        return triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets

    def validate_and_fix(n, colors, time_limit):
        """
        Takes a candidate coloring for n vertices and tries to fix it to be valid.
        Returns fixed colors or None if it fails within time_limit.
        """
        if n < 4:
            return colors
            
        triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets = precompute_sets(n)
        num_triples = len(triples)
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        # Ensure colors length matches
        if len(colors) != num_triples:
            return None
            
        colors = colors[:]
        
        # red_count_4[i] = number of red triples in 4-set i
        # blue_count_5[j] = number of blue triples in 5-set j
        red_count_4 = [0] * num_4sets
        blue_count_5 = [0] * num_5sets
        
        # Initialize counts
        for t_idx, t in enumerate(triples):
            is_red = (colors[t_idx] == 0)
            for s4_idx in triple_to_4sets[t_idx]:
                if is_red:
                    red_count_4[s4_idx] += 1
            for s5_idx in triple_to_5sets[t_idx]:
                if not is_red: # blue
                    blue_count_5[s5_idx] += 1
                    
        def calc_cost():
            cost = 0
            for c in red_count_4:
                if c == 4:
                    cost += 1
            for c in blue_count_5:
                if c == 10:
                    cost += 1
            return cost
            
        current_cost = calc_cost()
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors[:]
        
        # Conflict-driven local search
        # We maintain a list of "hot" triples that are part of violations.
        # However, simple random sampling with strong bias towards violation members is effective.
        
        # To speed up, we can identify which triples are in violations and sample from them more often.
        # But recomputing the list of violating triples is expensive if done every step.
        # Instead, we use a hybrid: 
        # 1. If cost is high, pick a random violating 4-set or 5-set, and flip one of its triples.
        # 2. If cost is low, do standard random flips.
        
        temp = 1.0
        cooling_rate = 0.9995
        step = 0
        
        while time.time() < time_limit:
            step += 1
            if temp < 0.01:
                break
                
            # Strategy: Pick a move that is likely to reduce cost
            # Find a random violating 4-set or 5-set if they exist
            move_t_idx = None
            old_color = None
            
            # Probabilistically choose to target a violation
            if current_cost > 0 and random.random() < 0.7:
                # Try to find a violating 4-set
                # We can't iterate all 4-sets every step if n is large (34C4 ~ 46376), 
                # but we can maintain a set of violating indices? 
                # For simplicity and speed in Python, we'll sample a few random 4-sets/5-sets to find a violation.
                # If we find one, we pick a random triple within it to flip.
                
                found_violation = False
                # Sample up to 10 random 4-sets
                for _ in range(10):
                    s4_idx = random.randint(0, num_4sets - 1)
                    if red_count_4[s4_idx] == 4:
                        # Found a red K4
                        s4 = four_sets[s4_idx]
                        # Pick one of the 4 triples to flip
                        # The triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
                        a,b,c,d = s4
                        t_candidates = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
                        t_to_flip = random.choice(t_candidates)
                        move_t_idx = triple_index[t_to_flip]
                        old_color = colors[move_t_idx]
                        found_violation = True
                        break
                        
                if not found_violation:
                    # Sample up to 10 random 5-sets
                    for _ in range(10):
                        s5_idx = random.randint(0, num_5sets - 1)
                        if blue_count_5[s5_idx] == 10:
                            # Found a blue K5
                            s5 = five_sets[s5_idx]
                            # Pick one of the 10 triples to flip
                            t_candidates = list(combinations(s5, 3))
                            t_to_flip = random.choice(t_candidates)
                            move_t_idx = triple_index[t_to_flip]
                            old_color = colors[move_t_idx]
                            found_violation = True
                            break
                            
            if not found_violation or move_t_idx is None:
                # Fallback to random triple
                move_t_idx = random.randint(0, num_triples - 1)
                old_color = colors[move_t_idx]
                
            new_color = 1 - old_color
            
            # Calculate delta cost
            delta_cost = 0
            affected_4 = triple_to_4sets[move_t_idx]
            affected_5 = triple_to_5sets[move_t_idx]
            
            if old_color == 0: # Red -> Blue
                for s4_idx in affected_4:
                    old_c = red_count_4[s4_idx]
                    new_c = old_c - 1
                    if old_c == 4: delta_cost -= 1
                    if new_c == 4: delta_cost += 1
                for s5_idx in affected_5:
                    old_c = blue_count_5[s5_idx]
                    new_c = old_c + 1
                    if old_c == 10: delta_cost -= 1
                    if new_c == 10: delta_cost += 1
                    
            else: # Blue -> Red
                for s4_idx in affected_4:
                    old_c = red_count_4[s4_idx]
                    new_c = old_c + 1
                    if old_c == 4: delta_cost -= 1
                    if new_c == 4: delta_cost += 1
                for s5_idx in affected_5:
                    old_c = blue_count_5[s5_idx]
                    new_c = old_c - 1
                    if old_c == 10: delta_cost -= 1
                    if new_c == 10: delta_cost += 1
            
            # Accept if delta < 0 or with probability exp(-delta/temp)
            accept = False
            if delta_cost < 0:
                accept = True
            elif delta_cost == 0:
                # Accept neutral moves to explore, but maybe less often
                if random.random() < 0.5:
                    accept = True
            else:
                if temp > 0:
                    if random.random() < math.exp(-delta_cost / temp):
                        accept = True
            
            if accept:
                colors[move_t_idx] = new_color
                for s4_idx in affected_4:
                    if old_color == 0:
                        red_count_4[s4_idx] -= 1
                    else:
                        red_count_4[s4_idx] += 1
                for s5_idx in affected_5:
                    if old_color == 0:
                        blue_count_5[s5_idx] += 1
                    else:
                        blue_count_5[s5_idx] -= 1
                
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                pass
                
            temp *= cooling_rate
            
        if best_cost == 0:
            return best_colors
        return None

    def generate_algebraic_coloring(n):
        """
        Generates a structured coloring using GF(2^5) logic.
        This provides a good starting point with low initial cost.
        """
        # GF(32) construction
        # Let V = GF(32). 
        # Color triple (a,b,c) red if a+b+c is a square in GF(32) (or some other quadratic form condition).
        # Actually, standard Ramsey constructions often use:
        # Red if (a-b)(b-c)(c-a) is a square in the field.
        
        # GF(2^5) with polynomial x^5 + x^2 + 1
        # Elements 0..31
        
        # Precompute squares in GF(32)
        # In char 2, map x -> x^2 is a bijection.
        # Let's define a quadratic form Q(x) = x^2 + x.
        # Or simpler: use the trace.
        
        # Let's try a simpler heuristic that mimics algebraic structure:
        # Color (a,b,c) red if (a^2 + b^2 + c^2 + ab + bc + ca) % 2 == 0 ?
        # In GF(2), a^2+a = a^2+a.
        
        # Let's just use a random seed but with a bias towards avoiding immediate K4s?
        # No, let's stick to the prompt's inspiration: "deterministic, symmetry-based construction using a finite field".
        
        # GF(32) multiplication table
        poly = 0b100101 # 37? No, x^5 + x^2 + 1 is 1 0010 1? 
        # x^5 + x^2 + 1 -> bits: 100101 (37). 
        # Wait, GF(32) needs a primitive polynomial of degree 5.
        # x^5 + x^2 + 1 is primitive.
        
        def gf_mult(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                a <<= 1
                if a & 32:
                    a ^= 0b100101
                b >>= 1
            return res
        
        # Precompute multiplication table
        mult_table = [[0]*32 for _ in range(32)]
        for i in range(32):
            for j in range(32):
                mult_table[i][j] = gf_mult(i, j)
                
        # In GF(2^5), the set of squares is the entire field (since char 2).
        # So "square" is not a distinguishing property.
        # We need a quadratic form.
        # Let Q(x) = x^2 + x.
        # Q is a linear map in GF(2^n)? No, x^2 is linear, x is linear. So Q is linear.
        # We need a non-degenerate quadratic form.
        # Q(x) = x^2 + x^3 ?
        
        # Let's use a simpler heuristic: 
        # Color (a,b,c) red if (a+b+c) is in a specific subset S of GF(32).
        # To avoid monochromatic K4, we need balance.
        
        # Actually, the most robust "algebraic" start for this specific task in Python without heavy libs is:
        # Use a random coloring but initialized with a "checkerboard" like pattern on the indices?
        
        # Let's go back to the "Inspiration 3" which used GF(31) (prime field).
        # GF(31) is a prime field.
        # Color (a,b,c) red if (a-b)(b-c)(c-a) is a quadratic residue mod 31.
        
        # Quadratic residues mod 31:
        # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25, 6^2=36=5, 7^2=49=18, 8^2=64=2, 9^2=81=19, 10^2=100=7, 11^2=121=28, 12^2=144=20, 13^2=169=14, 14^2=196=10, 15^2=225=8
        # Residues: {1, 2, 4, 5, 7, 8, 9, 10, 14, 16, 18, 19, 20, 25, 28}
        
        residues = set()
        for i in range(1, 16):
            residues.add((i*i) % 31)
            
        colors = []
        for t in combinations(range(n), 3):
            a, b, c = t
            val = ((a - b) * (b - c) * (c - a)) % 31
            if val in residues:
                colors.append(0) # Red
            else:
                colors.append(1) # Blue
                
        return colors

    # Try n from 34 downwards
    # For n=34, the algebraic construction (mod 31) doesn't directly apply to 34 vertices.
    # We can map 0..30 to GF(31) elements, and 31,32,33 to "infinity" or just randomize them?
    # Better: Use the GF(31) construction for the first 31 vertices, and random for the last 3,
    # then let the local search fix everything.
    
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        
        # Allocate time based on difficulty
        # n=34 is hard. n=31 is easier.
        if n == 34:
            time_limit = time.time() + 40.0
        elif n == 33:
            time_limit = time.time() + 30.0
        elif n == 32:
            time_limit = time.time() + 20.0
        elif n == 31:
            time_limit = time.time() + 15.0
        else:
            time_limit = time.time() + 5.0
            
        if time_limit > deadline:
            time_limit = deadline
            
        # Generate initial coloring
        if n <= 31:
            # Direct algebraic construction
            initial_colors = generate_algebraic_coloring(n)
        else:
            # Extend algebraic construction
            # Use GF(31) for 0..30
            base_colors = generate_algebraic_coloring(31)
            # For vertices 31..n-1, we need to define triples involving them.
            # This is tricky with the simple mod 31 formula.
            # Let's just randomize the whole thing for n>31, but seed with the structure?
            # Or better: Random init for n>31 is fine, local search is strong.
            # However, a structured init is better.
            # Let's create a random coloring for n, but bias it?
            # Actually, for n=34, random init + strong local search is the way to go.
            # The "algebraic" part is mainly for n=31 where it's known to work well.
            initial_colors = [random.randint(0, 1) for _ in range(len(get_triples(n)))]
            
        res = validate_and_fix(n, initial_colors, time_limit)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback: return a small valid coloring
    n = 4
    colors = [0, 0, 0, 1]
    return {'n': n, 'colors': colors}

# For testing
if __name__ == "__main__":
    res = construct()
    print(res['n'])
