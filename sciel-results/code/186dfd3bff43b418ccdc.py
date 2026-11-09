"""Covering design C(v,k,t) — Genetic Algorithm with Local Search."""
import itertools
import time
import random
import copy

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 110.0   # Safety margin
    
    points = list(range(v))
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 3. Precompute block -> t-subsets mapping
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)
        
    # 4. Precompute t-subset -> blocks mapping
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    def run_greedy(randomize=False, deadline=None):
        """Greedy set cover."""
        if deadline is None:
            deadline = DEADLINE
        block_gain = [len(subs) for subs in block_tsubs]
        is_covered = [0] * N_t
        uncovered_count = N_t
        selected_blocks = []
        
        while uncovered_count > 0 and time.time() < deadline:
            best_block_idx = -1
            max_gain = -1
            candidates = []
            
            if randomize:
                for i in range(N_blocks):
                    if block_gain[i] > max_gain:
                        max_gain = block_gain[i]
                        candidates = [i]
                    elif block_gain[i] == max_gain:
                        candidates.append(i)
                
                if not candidates:
                    break
                
                best_block_idx = random.choice(candidates)
            else:
                for i in range(N_blocks):
                    if block_gain[i] > max_gain:
                        max_gain = block_gain[i]
                        best_block_idx = i
            
            if best_block_idx == -1 or max_gain == 0:
                break
                
            selected_blocks.append(all_blocks[best_block_idx])
            
            for si in block_tsubs[best_block_idx]:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    for b_idx in tsub_blocks[si]:
                        block_gain[b_idx] -= 1
                        
        return selected_blocks

    def repair_cover(blocks, deadline=None):
        """Ensure the cover is valid by adding blocks if necessary."""
        if deadline is None:
            deadline = DEADLINE
        
        covered = [0] * N_t
        for b in blocks:
            for s in itertools.combinations(b, t):
                covered[t_idx[s]] = 1
        
        uncovered_set = set(i for i, c in enumerate(covered) if c == 0)
        if not uncovered_set:
            return blocks
        
        # Calculate initial gain
        block_gain = [0] * N_blocks
        for ui in uncovered_set:
            for b_idx in tsub_blocks[ui]:
                block_gain[b_idx] += 1
        
        while uncovered_set and time.time() < deadline:
            best_block_idx = -1
            max_gain = -1
            for i in range(N_blocks):
                if block_gain[i] > max_gain:
                    max_gain = block_gain[i]
                    best_block_idx = i
            
            if max_gain == 0:
                break
            
            new_block = all_blocks[best_block_idx]
            blocks.append(new_block)
            
            # Update coverage
            newly_covered = []
            for si in block_tsubs[best_block_idx]:
                if si in uncovered_set:
                    covered[si] = 1
                    uncovered_set.remove(si)
                    newly_covered.append(si)
            
            # Update gains
            for si in newly_covered:
                for b_idx in tsub_blocks[si]:
                    block_gain[b_idx] -= 1

        return blocks

    def try_reduce(blocks, deadline=None):
        """Remove redundant blocks."""
        if deadline is None:
            deadline = DEADLINE
        if not blocks:
            return blocks
        
        block_subs = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(t_idx[s])
            block_subs.append(subs)
            
        n = len(blocks)
        cov = [0] * N_t
        for subs in block_subs:
            for si in subs:
                cov[si] += 1
                
        removed = [False] * n
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                removable = True
                for si in block_subs[i]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    removed[i] = True
                    changed = True
                    for si in block_subs[i]:
                        cov[si] -= 1
                        
        return [b for i, b in enumerate(blocks) if not removed[i]]

    def is_valid_cover(blocks):
        """Check if blocks form a valid cover."""
        if not blocks:
            return N_t == 0
        covered = [0] * N_t
        for b in blocks:
            for s in itertools.combinations(b, t):
                covered[t_idx[s]] = 1
        return all(covered)

    def evaluate_cover(blocks):
        """Return number of blocks (lower is better)."""
        return len(blocks)

    # --- Genetic Algorithm ---
    
    # Initialize population with greedy solutions
    pop_size = 20
    population = []
    
    # First few are deterministic/randomized greedy
    for i in range(min(5, pop_size)):
        sol = run_greedy(randomize=(i > 0), deadline=DEADLINE)
        sol = repair_cover(sol, deadline=DEADLINE)
        sol = try_reduce(sol, deadline=DEADLINE)
        population.append(sol)
    
    # Fill rest with more randomized greedy
    while len(population) < pop_size and time.time() < DEADLINE - 30:
        sol = run_greedy(randomize=True, deadline=DEADLINE)
        sol = repair_cover(sol, deadline=DEADLINE)
        sol = try_reduce(sol, deadline=DEADLINE)
        population.append(sol)
    
    best_sol = min(population, key=len)
    best_len = len(best_sol)
    
    # Genetic algorithm loop
    gen = 0
    while time.time() < DEADLINE - 10:
        gen += 1
        
        # Evaluate population
        scored = sorted(population, key=len)
        
        # Elitism: keep top half
        elite_count = max(1, pop_size // 2)
        new_population = [copy.deepcopy(s) for s in scored[:elite_count]]
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < DEADLINE - 5:
            # Select parents via tournament selection
            def tournament_select():
                candidates = random.sample(population, min(3, len(population)))
                return min(candidates, key=len)
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover: combine blocks from both parents, then reduce
            combined = list(set(parent1) | set(parent2))
            # Shuffle to break symmetry
            random.shuffle(combined)
            
            # Try to reduce
            offspring = repair_cover(combined, deadline=min(DEADLINE - 2, time.time() + 5))
            offspring = try_reduce(offspring, deadline=min(DEADLINE - 2, time.time() + 5))
            
            if len(offspring) <= best_len:
                new_population.append(offspring)
            else:
                # Mutation: replace some blocks with random ones
                mutated = list(parent1)
                if mutated:
                    for _ in range(random.randint(1, 3)):
                        if mutated:
                            idx = random.randint(0, len(mutated) - 1)
                            # Try to find a better replacement
                            # Remove block at idx and see if still valid
                            candidate = [b for i, b in enumerate(mutated) if i != idx]
                            if is_valid_cover(candidate):
                                mutated = candidate
                            else:
                                # Replace with random block
                                new_block = random.choice(all_blocks)
                                mutated[idx] = new_block
                                # Reduce
                                mutated = repair_cover(mutated, deadline=min(DEADLINE - 2, time.time() + 2))
                                mutated = try_reduce(mutated, deadline=min(DEADLINE - 2, time.time() + 2))
                new_population.append(mutated)
        
        # Update population
        population = new_population[:pop_size]
        
        # Update best
        current_best = min(population, key=len)
        if len(current_best) < best_len:
            best_len = len(current_best)
            best_sol = copy.deepcopy(current_best)
    
    # Final reduction
    best_sol = repair_cover(best_sol, deadline=DEADLINE)
    best_sol = try_reduce(best_sol, deadline=DEADLINE)
    
    return best_sol
