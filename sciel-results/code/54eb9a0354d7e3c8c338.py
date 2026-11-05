import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    all_triples = list(itertools.combinations(range(v), t))
    triple_index = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Cache for block -> set of triple indices
    block_cache = {}
    
    def get_covered_triples(block):
        if block in block_cache:
            return block_cache[block]
        # block is a tuple
        res = frozenset(triple_index[triple] for triple in itertools.combinations(block, t))
        block_cache[block] = res
        return res

    def get_blocks_containing(triple):
        remaining_points = [p for p in range(v) if p not in triple]
        for combo in itertools.combinations(remaining_points, k - t):
            yield tuple(sorted(triple + combo))

    best_blocks = None
    best_count = float('inf')
    
    def greedy_cover(uncovered_set, existing_blocks=None):
        if existing_blocks is None:
            existing_blocks = []
        else:
            existing_blocks = list(existing_blocks)
            
        current_uncovered = set(uncovered_set)
        current_blocks = existing_blocks
        
        while current_uncovered:
            # Pick a random uncovered triple
            target_triple_idx = random.choice(list(current_uncovered))
            target_triple = all_triples[target_triple_idx]
            
            best_block = None
            best_gain = -1
            
            for block in get_blocks_containing(target_triple):
                # Calculate gain
                covered = get_covered_triples(block)
                # Intersection with current_uncovered
                gain = len(covered & current_uncovered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
            
            current_blocks.append(best_block)
            
            # Update uncovered
            covered_by_new = get_covered_triples(best_block)
            current_uncovered -= covered_by_new
            
            if time.time() > deadline - 5:
                break
        
        return current_blocks, current_uncovered

    def remove_redundant(blocks):
        if not blocks:
            return blocks
            
        triple_counts = {}
        for b in blocks:
            covered = get_covered_triples(b)
            for idx in covered:
                triple_counts[idx] = triple_counts.get(idx, 0) + 1
        
        redundant_indices = set()
        for i, b in enumerate(blocks):
            covered = get_covered_triples(b)
            is_redundant = True
            for idx in covered:
                if triple_counts.get(idx, 0) == 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.add(i)
        
        if redundant_indices:
            return [b for i, b in enumerate(blocks) if i not in redundant_indices]
        return blocks

    # Initial Greedy
    blocks, _ = greedy_cover(set(range(num_triples)))
    blocks = remove_redundant(blocks)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Iterated Improvement
    while time.time() < deadline - 10:
        if not best_blocks:
            continue
            
        num_to_remove = random.randint(1, 2)
        if len(best_blocks) <= num_to_remove:
            continue
            
        perturbed_blocks = list(best_blocks)
        indices_to_remove = random.sample(range(len(perturbed_blocks)), num_to_remove)
        for idx in sorted(indices_to_remove, reverse=True):
            perturbed_blocks.pop(idx)
        
        # Calculate uncovered
        covered_triples = set()
        for b in perturbed_blocks:
            covered_triples.update(get_covered_triples(b))
        
        current_uncovered = set(range(num_triples)) - covered_triples
        
        if not current_uncovered:
            improved = remove_redundant(perturbed_blocks)
            if len(improved) < best_count:
                best_blocks = improved
                best_count = len(best_blocks)
            continue
            
        new_blocks, remaining_uncovered = greedy_cover(current_uncovered, perturbed_blocks)
        
        if not remaining_uncovered:
            new_blocks = remove_redundant(new_blocks)
            if len(new_blocks) < best_count:
                best_blocks = new_blocks
                best_count = len(new_blocks)
                
    return [list(b) for b in best_blocks]
