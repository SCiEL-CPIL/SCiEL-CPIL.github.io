import itertools
import time
import random
import math

def construct(q, n, radius):
    # For the specific task: q=3, n=6, radius=1
    # Use a multi-phase local search with greedy initialization and aggressive reduction.
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (sphere of radius 1)
    neighbors = []
    for w in all_words:
        w_list = list(w)
        nbrs = [word_to_idx[w]]  # include self
        for i in range(n):
            orig_val = w_list[i]
            for v in range(q):
                if v != orig_val:
                    w_list[i] = v
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(nbrs)
    
    # Helper: compute coverage counts from a list of codeword indices
    def compute_coverage(code_indices):
        coverage = [0] * total_words
        for ci in code_indices:
            for nbr in neighbors[ci]:
                coverage[nbr] += 1
        return coverage
    
    # Helper: count uncovered
    def count_uncovered(coverage):
        return sum(1 for c in coverage if c == 0)
    
    # Phase 1: Greedy maximal code
    # Start with all words, greedily select codewords to cover everything
    # Actually, let's start from a smaller set and add as needed
    
    # Better: start with a random set, then reduce
    rng = random.Random(123)
    
    # Initial: tetracode-based direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    code_indices = [word_to_idx[tuple(cw)] for cw in initial_code]
    coverage = compute_coverage(code_indices)
    uncovered_count = count_uncovered(coverage)
    
    # Also try a smaller random initial set
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        # Try a random subset of size around 73-80
        target_size = 73 + trial * 2
        random_indices = rng.sample(range(total_words), target_size)
        random_indices_set = set(random_indices)
        code_indices2 = list(random_indices_set)
        coverage2 = compute_coverage(code_indices2)
        uncovered2 = count_uncovered(coverage2)
        
        # Keep the best initial
        if uncovered2 < uncovered_count or (uncovered2 == uncovered_count and len(code_indices2) < len(code_indices)):
            code_indices = code_indices2
            coverage = coverage2
            uncovered_count = uncovered2
    
    best_code = list(code_indices)
    best_uncovered = uncovered_count
    best_size = len(code_indices)
    
    # Phase 2: Aggressive reduction - repeatedly remove redundant codewords
    def try_reduce(code_indices, coverage):
        """Try to remove redundant codewords. Returns (new_code, new_coverage, changed)."""
        changed = False
        # Sort by coverage contribution (remove low-contribution ones first)
        # A codeword is redundant if all its neighbors have coverage >= 2
        for i in range(len(code_indices) - 1, -1, -1):
            ci = code_indices[i]
            is_redundant = True
            for nbr in neighbors[ci]:
                if coverage[nbr] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                code_indices.pop(i)
                for nbr in neighbors[ci]:
                    coverage[nbr] -= 1
                changed = True
        return code_indices, coverage, changed
    
    # Phase 3: Local search with swaps
    T = 2.0
    T_min = 0.05
    cooling = 0.9995
    
    iterations = 0
    while time.time() < deadline - 5.0:
        iterations += 1
        
        # First, always try to reduce
        code_indices, coverage, changed = try_reduce(code_indices, coverage)
        if changed:
            uncovered_count = count_uncovered(coverage)
            if uncovered_count == 0 and len(code_indices) < best_size:
                best_size = len(code_indices)
                best_code = list(code_indices)
                best_uncovered = 0
        
        if uncovered_count == 0 and best_size <= 73:
            break
        
        # Random move
        move_type = rng.random()
        
        if move_type < 0.35 and len(code_indices) > 1:
            # Remove a random codeword
            idx_to_remove = rng.randrange(len(code_indices))
            ci = code_indices[idx_to_remove]
            
            delta_uncovered = 0
            for nbr in neighbors[ci]:
                if coverage[nbr] == 1:
                    delta_uncovered += 1
            
            # Apply
            for nbr in neighbors[ci]:
                coverage[nbr] -= 1
            code_indices.pop(idx_to_remove)
            new_uncovered = uncovered_count + delta_uncovered
            
            # Accept if better or with probability
            if new_uncovered < best_uncovered or (new_uncovered == best_uncovered and len(code_indices) < best_size):
                uncovered_count = new_uncovered
                if new_uncovered < best_uncovered or len(code_indices) < best_size:
                    best_uncovered = new_uncovered
                    best_size = len(code_indices)
                    best_code = list(code_indices)
            elif new_uncovered <= best_uncovered:
                delta = new_uncovered - best_uncovered
                if delta == 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                    uncovered_count = new_uncovered
                else:
                    # Reject
                    for nbr in neighbors[ci]:
                        coverage[nbr] += 1
                    code_indices.insert(idx_to_remove, ci)
            else:
                # Reject
                for nbr in neighbors[ci]:
                    coverage[nbr] += 1
                    code_indices.insert(idx_to_remove, ci)
        
        elif move_type < 0.65:
            # Add a random codeword that covers uncovered words
            # Find uncovered words
            uncovered_words = [i for i in range(total_words) if coverage[i] == 0]
            if uncovered_words:
                # Pick a random uncovered word, and add a random neighbor as codeword
                target = rng.choice(uncovered_words)
                # Candidates: all neighbors of target
                candidates = neighbors[target]
                new_ci = rng.choice(candidates)
            else:
                new_ci = rng.randrange(total_words)
            
            if new_ci in code_indices:
                continue
            
            delta_uncovered = 0
            for nbr in neighbors[new_ci]:
                if coverage[nbr] == 0:
                    delta_uncovered -= 1
            
            for nbr in neighbors[new_ci]:
                coverage[nbr] += 1
            code_indices.append(new_ci)
            new_uncovered = uncovered_count + delta_uncovered
            
            if new_uncovered < best_uncovered or (new_uncovered == best_uncovered and len(code_indices) < best_size):
                uncovered_count = new_uncovered
                if new_uncovered < best_uncovered or len(code_indices) < best_size:
                    best_uncovered = new_uncovered
                    best_size = len(code_indices)
                    best_code = list(code_indices)
            elif new_uncovered <= best_uncovered:
                delta = new_uncovered - best_uncovered
                if delta == 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                    uncovered_count = new_uncovered
                else:
                    for nbr in neighbors[new_ci]:
                        coverage[nbr] -= 1
                    code_indices.pop()
            else:
                for nbr in neighbors[new_ci]:
                    coverage[nbr] -= 1
                code_indices.pop()
        
        else:
            # Swap: remove one, add another
            if len(code_indices) < 1:
                continue
            idx_to_remove = rng.randrange(len(code_indices))
            ci_remove = code_indices[idx_to_remove]
            
            # Find uncovered words to target
            uncovered_words = [i for i in range(total_words) if coverage[i] == 0]
            if uncovered_words:
                target = rng.choice(uncovered_words)
                # Candidates that cover target and are not already in code
                candidates = [nbr for nbr in neighbors[target] if nbr not in code_indices]
                if candidates:
                    new_ci = rng.choice(candidates)
                else:
                    new_ci = rng.randrange(total_words)
            else:
                new_ci = rng.randrange(total_words)
            
            if new_ci in code_indices:
                continue
            
            # Compute delta
            delta_remove = 0
            for nbr in neighbors[ci_remove]:
                if coverage[nbr] == 1:
                    delta_remove += 1
            
            delta_add = 0
            for nbr in neighbors[new_ci]:
                effective = coverage[nbr]
                if nbr in neighbors[ci_remove]:
                    effective -= 1
                if effective == 0:
                    delta_add -= 1
            
            total_delta = delta_remove + delta_add
            new_uncovered = uncovered_count + total_delta
            
            # Apply
            for nbr in neighbors[ci_remove]:
                coverage[nbr] -= 1
            for nbr in neighbors[new_ci]:
                coverage[nbr] += 1
            code_indices[idx_to_remove] = new_ci
            
            if new_uncovered < best_uncovered or (new_uncovered == best_uncovered and len(code_indices) < best_size):
                uncovered_count = new_uncovered
                if new_uncovered < best_uncovered or len(code_indices) < best_size:
                    best_uncovered = new_uncovered
                    best_size = len(code_indices)
                    best_code = list(code_indices)
            elif new_uncovered <= best_uncovered:
                delta = new_uncovered - best_uncovered
                if delta == 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                    uncovered_count = new_uncovered
                else:
                    for nbr in neighbors[new_ci]:
                        coverage[nbr] -= 1
                    for nbr in neighbors[ci_remove]:
                        coverage[nbr] += 1
                    code_indices[idx_to_remove] = ci_remove
            else:
                for nbr in neighbors[new_ci]:
                    coverage[nbr] -= 1
                for nbr in neighbors[ci_remove]:
                    coverage[nbr] += 1
                code_indices[idx_to_remove] = ci_remove
        
        T = max(T * cooling, T_min)
    
    # Final: use best_code
    final_code = list(best_code)
    
    # Final reduction pass
    if time.time() < deadline - 2.0:
        final_coverage = compute_coverage(final_code)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(len(final_code) - 1, -1, -1):
                ci = final_code[i]
                is_redundant = True
                for nbr in neighbors[ci]:
                    if final_coverage[nbr] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    final_code.pop(i)
                    for nbr in neighbors[ci]:
                        final_coverage[nbr] -= 1
                    changed = True
                    break
    
    # Verify
    result = [list(all_words[i]) for i in final_code]
    covered = set()
    for cw in result:
        idx = word_to_idx[tuple(cw)]
        covered.update(neighbors[idx])
    
    if len(covered) != total_words:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    return result
