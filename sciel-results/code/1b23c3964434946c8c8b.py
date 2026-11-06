import time
import math
import random
import sys
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def elapsed():
        return time.time() - start

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g; B //= g; C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A; B = -B; C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None or key in seen:
                    continue
                seen.add(key)
                A, B, C = key
                cnt = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        cnt += 1
                if cnt >= 4:
                    return True
        return False

    def count_collinear_triples(pts):
        n = len(pts)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        cnt += 1
        return cnt

    def build_triple_structure(pts):
        n = len(pts)
        triples = []
        # point_conflicts[i] is a list of (j, k) such that (i, j, k) is a triple
        point_conflicts = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples.append((i, j, k))
                        point_conflicts[i].append((j, k))
                        point_conflicts[j].append((i, k))
                        point_conflicts[k].append((i, j))
        return triples, point_conflicts

    def greedy_independent_set_size(pts):
        """
        Fast greedy estimator for the size of the largest subset with no 3 collinear.
        This is an upper bound on the true maximum independent set size.
        We want to MINIMIZE this score, so we want to MINIMIZE the size of this greedy set.
        Wait, the score is the LARGEST subset. The greedy method finds A subset.
        If we want to minimize the true max subset, we should try to make the greedy subset small?
        No, the greedy subset is a valid independent set. So TrueMax >= GreedySize.
        Minimizing GreedySize is a good proxy for minimizing TrueMax.
        
        Strategy: 
        1. Order points by degree (number of triples they are in) descending.
        2. Iterate through points. If adding the point doesn't create a triple with existing points, add it.
        """
        n = len(pts)
        if n < 3:
            return n
        
        triples, point_conflicts = build_triple_structure(pts)
        
        # Order by degree descending
        degrees = [len(point_conflicts[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
        
        selected = []
        selected_set = set()
        
        for idx in order:
            ok = True
            for a, b in point_conflicts[idx]:
                if a in selected_set and b in selected_set:
                    ok = False
                    break
            if ok:
                selected.append(idx)
                selected_set.add(idx)
                
        return len(selected)

    def exact_score(pts):
        """
        Exact branch and bound for the largest subset with no 3 collinear.
        Used sparingly for final verification or when time permits.
        """
        n = len(pts)
        if n < 3:
            return n
        
        triples, point_conflicts = build_triple_structure(pts)
        
        # Order points by degree descending
        degrees = [len(point_conflicts[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
        
        ordered_conflicts = [point_conflicts[order[i]] for i in range(n)]
        
        best = [0]
        
        def bbb(step, cur_mask, cur_size):
            if elapsed() > DEADLINE - 5:
                return
            if cur_size > best[0]:
                best[0] = cur_size
            if step == n:
                return
            if cur_size + (n - step) <= best[0]:
                return
            
            idx = order[step]
            
            # Option 1: skip idx
            bbb(step + 1, cur_mask, cur_size)
            
            # Option 2: include idx if no conflict
            ok = True
            for a, b in ordered_conflicts[step]:
                if (cur_mask & (1 << a)) and (cur_mask & (1 << b)):
                    ok = False
                    break
            if ok:
                bbb(step + 1, cur_mask | (1 << idx), cur_size + 1)
                
        bbb(0, 0, 0)
        return best[0]

    def generate_initial_population(pop_size):
        pop = []
        rng = random.Random(42)
        for _ in range(pop_size):
            # Start with a random valid configuration
            pts = []
            used = set()
            while len(pts) < 20:
                p = (rng.randint(-100, 100), rng.randint(-100, 100))
                if p in used:
                    continue
                if has_4_collinear(pts + [p]):
                    continue
                pts.append(p)
                used.add(p)
            pop.append(pts)
        return pop

    def crossover(p1, p2, rng):
        # Uniform crossover
        child = []
        used = set()
        # Shuffle indices to pick from
        indices = list(range(20))
        rng.shuffle(indices)
        # Pick 10 from p1, 10 from p2
        p1_indices = set(indices[:10])
        p2_indices = set(indices[10:])
        
        # This simple crossover might create invalid configs (duplicates, 4-collinear)
        # We will repair later.
        # Actually, let's just mix coordinates.
        child = []
        for i in range(20):
            if rng.random() < 0.5:
                child.append(p1[i])
            else:
                child.append(p2[i])
        return child

    def mutate(ind, rng, mutation_rate=0.2):
        new_ind = ind[:]
        for i in range(20):
            if rng.random() < mutation_rate:
                # Perturb coordinates
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                new_x = max(-100, min(100, new_ind[i][0] + dx))
                new_y = max(-100, min(100, new_ind[i][1] + dy))
                new_ind[i] = (new_x, new_y)
        return new_ind

    def repair(ind):
        """
        Repair an individual to ensure:
        1. 20 distinct points
        2. |x|, |y| <= 100
        3. No 4 collinear
        """
        # 1. Distinct points
        seen = set()
        unique_pts = []
        for p in ind:
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
        
        # If we lost points, add random ones
        rng = random.Random()
        while len(unique_pts) < 20:
            p = (rng.randint(-100, 100), rng.randint(-100, 100))
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
        
        # 2. Bounds are handled by mutation/clamping, but let's clamp just in case
        for i in range(20):
            x, y = unique_pts[i]
            unique_pts[i] = (max(-100, min(100, x)), max(-100, min(100, y)))
        
        # 3. No 4 collinear
        # This is the hard part. We will try to remove points that form 4-collinear sets
        # and replace them with random points.
        # Simple heuristic: Check all lines. If a line has >= 4 points, remove one and replace.
        # To avoid infinite loops, limit repairs.
        for _ in range(100):
            if not has_4_collinear(unique_pts):
                break
            
            # Find a 4-collinear set
            n = len(unique_pts)
            removed = False
            for i in range(n):
                for j in range(i+1, n):
                    key = line_key(unique_pts[i], unique_pts[j])
                    if key is None: continue
                    A, B, C = key
                    cnt = 0
                    for k in range(n):
                        x3, y3 = unique_pts[k]
                        if A * x3 + B * y3 + C == 0:
                            cnt += 1
                    if cnt >= 4:
                        # Remove a random point from this line
                        # Find indices on this line
                        line_pts = []
                        for k in range(n):
                            x3, y3 = unique_pts[k]
                            if A * x3 + B * y3 + C == 0:
                                line_pts.append(k)
                        # Pick one to remove
                        idx_to_remove = line_pts[rng.randint(0, len(line_pts)-1)]
                        # Replace with random point
                        while True:
                            new_p = (rng.randint(-100, 100), rng.randint(-100, 100))
                            if new_p not in seen:
                                break
                        seen.remove(unique_pts[idx_to_remove])
                        seen.add(new_p)
                        unique_pts[idx_to_remove] = new_p
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                # If we couldn't fix it with simple replacement, restart this individual
                # This should be rare
                unique_pts = generate_initial_population(1)[0]
                break
                
        return unique_pts

    def fitness(ind):
        """
        Higher is better.
        We want to maximize triples.
        """
        if has_4_collinear(ind):
            return -10000
        triples = count_collinear_triples(ind)
        # Greedy independent set size (lower is better for score, so we subtract it)
        # Let's just use triples for now, as it's a strong signal.
        # Or combine: triples - greedy_size
        greedy_size = greedy_independent_set_size(ind)
        return triples * 10 - greedy_size

    # Main GA Loop
    pop_size = 50
    pop = generate_initial_population(pop_size)
    
    # Evaluate initial pop
    for i in range(pop_size):
        pop[i] = repair(pop[i])
    
    best_ind = None
    best_fit = -100000
    best_score = 20 # Max possible score is 20
    
    # Evaluate initial population
    for ind in pop:
        fit = fitness(ind)
        if fit > best_fit:
            best_fit = fit
            best_ind = ind[:]
            # Calculate actual score for tracking
            # Use greedy for speed
            s = greedy_independent_set_size(best_ind)
            if s < best_score:
                best_score = s
                
    generation = 0
    while elapsed() < DEADLINE - 10:
        generation += 1
        new_pop = []
        
        # Elitism
        pop_sorted = sorted(pop, key=lambda x: fitness(x), reverse=True)
        new_pop.append(pop_sorted[0][:])
        if pop_size > 1:
            new_pop.append(pop_sorted[1][:])
            
        # Selection and Crossover
        while len(new_pop) < pop_size:
            # Tournament selection
            p1 = random.choice(pop)
            p2 = random.choice(pop)
            if fitness(p1) < fitness(p2):
                p1, p2 = p2, p1
            
            child = crossover(p1, p2, random)
            child = mutate(child, random, 0.3)
            child = repair(child)
            new_pop.append(child)
            
        pop = new_pop
        
        # Update best
        for ind in pop:
            fit = fitness(ind)
            if fit > best_fit:
                best_fit = fit
                best_ind = ind[:]
                s = greedy_independent_set_size(best_ind)
                if s < best_score:
                    best_score = s

    # Local Search on best_ind
    current_ind = best_ind[:]
    current_fit = fitness(current_ind)
    current_score = greedy_independent_set_size(current_ind)
    
    rng = random.Random(123)
    T = 5.0
    T_min = 0.1
    cooling = 0.99
    
    while elapsed() < DEADLINE - 2:
        # Pick a random point to move
        idx = rng.randint(0, 19)
        old_p = current_ind[idx]
        new_p = (max(-100, min(100, old_p[0] + rng.randint(-3, 3))),
                 max(-100, min(100, old_p[1] + rng.randint(-3, 3))))
        
        if new_p in current_ind:
            continue
            
        trial_ind = current_ind[:]
        trial_ind[idx] = new_p
        
        if has_4_collinear(trial_ind):
            continue
            
        trial_fit = fitness(trial_ind)
        trial_score = greedy_independent_set_size(trial_ind)
        
        # Accept if better fitness or with probability exp((new-old)/T)
        if trial_fit > current_fit or (trial_fit == current_fit and trial_score < current_score) or rng.random() < math.exp((trial_fit - current_fit) / T):
            current_ind = trial_ind
            current_fit = trial_fit
            current_score = trial_score
            
            if current_score < best_score:
                best_score = current_score
                best_ind = current_ind[:]
                
        T *= cooling
        if T < T_min:
            T = T_min

    # Final exact score check if time permits
    if elapsed() < DEADLINE - 5:
        final_score = exact_score(best_ind)
        if final_score < best_score:
            best_score = final_score

    return [[x, y] for x, y in best_ind]

if __name__ == "__main__":
    pts = construct()
    print(pts)
