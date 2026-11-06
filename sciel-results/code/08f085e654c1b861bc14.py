import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(16))
    all_triples_list = list(itertools.combinations(all_points, 3))
    num_triples = len(all_triples_list)
    
    # Precompute masks and triple-to-index mapping
    triple_masks = {}
    triple_to_idx = {}
    for i, t in enumerate(all_triples_list):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[t] = mask
        triple_to_idx[t] = i
        
    # Precompute conflict bitsets
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[all_triples_list[i]]
        for j in range(i+1, num_triples):
            mj = triple_masks[all_triples_list[j]]
            inter = mi & mj
            if inter & (inter - 1):  # share > 1 point
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    # Precompute for each triple, the set of pairs it covers
    triple_pairs = {}
    for i, t in enumerate(all_triples_list):
        pairs = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
        triple_pairs[i] = pairs
    
    # Precompute point-to-triples mapping
    point_to_triples = [[] for _ in range(16)]
    for i, t in enumerate(all_triples_list):
        for p in t:
            point_to_triples[p].append(i)
    
    # Precompute for each triple, its color pattern under a 2-coloring
    # A triple is "monochromatic" if all 3 points have the same color
    # We'll use this to bias triple selection
    
    def find_max_independent_set(active_mask):
        """
        Find the size of the largest subset of points that contains no triple.
        Uses a branch-and-bound approach.
        """
        if active_mask == 0:
            return 16
        
        # Get list of active triple masks
        active_masks = []
        active_indices = []
        for i in range(num_triples):
            if (active_mask >> i) & 1:
                active_masks.append(triple_masks[all_triples_list[i]])
                active_indices.append(i)
        
        if not active_masks:
            return 16
        
        best_k = 0
        
        # Check k from 16 down to 1
        for k in range(16, 0, -1):
            found = False
            for subset in itertools.combinations(all_points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                is_ind = True
                for tm in active_masks:
                    if (tm & smask) == tm:
                        is_ind = False
                        break
                if is_ind:
                    best_k = k
                    found = True
                    break
            if found:
                break
            if time.time() > deadline - 5:
                return best_k if best_k > 0 else k
        return best_k if best_k > 0 else 1
    
    def simulate_annealing_coloring(num_iterations=500):
        """
        Use simulated annealing to find a 2-coloring of the 16 points that
        minimizes the maximum number of monochromatic triples in each color class.
        This provides a guide for triple selection.
        """
        # Initial random coloring: 0 or 1 for each point
        coloring = [random.randint(0, 1) for _ in range(16)]
        
        def count_mono_triples(col):
            """Count monochromatic triples for a given coloring."""
            mono_count = [0, 0]  # count for color 0 and color 1
            for t in all_triples_list:
                c0, c1, c2 = col[t[0]], col[t[1]], col[t[2]]
                if c0 == c1 == c2:
                    mono_count[c0] += 1
            return mono_count
        
        current_mono = count_mono_triples(coloring)
        current_score = max(current_mono[0], current_mono[1])
        
        best_coloring = coloring[:]
        best_score = current_score
        
        T = 5.0  # initial temperature
        cooling = 0.995
        
        for it in range(num_iterations):
            if time.time() > deadline - 30:
                break
            
            # Pick a random point to flip
            p = random.randint(0, 15)
            old_color = coloring[p]
            coloring[p] = 1 - old_color
            
            new_mono = count_mono_triples(coloring)
            new_score = max(new_mono[0], new_mono[1])
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_coloring = coloring[:]
            else:
                coloring[p] = old_color
            
            T *= cooling
        
        return best_coloring, best_score
    
    def generate_greedy_initial(coloring=None):
        """
        Greedily construct a linear triple system by adding triples that cover
        the most uncovered point-pairs, ensuring linearity.
        If a coloring is provided, bias towards triples that are NOT monochromatic
        (i.e., they have points of both colors), as these help break up independent sets.
        """
        active_mask = 0
        covered_pairs = set()
        remaining_triples = list(range(num_triples))
        
        # Precompute triple "quality" based on coloring
        triple_quality = [0] * num_triples
        if coloring is not None:
            for i, t in enumerate(all_triples_list):
                c0, c1, c2 = coloring[t[0]], coloring[t[1]], coloring[t[2]]
                # Mixed triples (not all same color) are more valuable
                if c0 == c1 == c2:
                    triple_quality[i] = 0  # monochromatic, less useful
                else:
                    triple_quality[i] = 1  # mixed, more useful
        
        while remaining_triples:
            if time.time() > deadline - 15:
                break
            
            # Find the triple that covers the most uncovered pairs, with bias from coloring
            best_idx = -1
            best_score = -1
            
            # Sample a subset for speed if too many remaining
            candidates = remaining_triples if len(remaining_triples) <= 100 else random.sample(remaining_triples, 100)
            
            for idx in candidates:
                if (conflict_bits[idx] & active_mask) != 0:
                    continue
                new_pairs = 0
                for pair in triple_pairs[idx]:
                    if pair not in covered_pairs:
                        new_pairs += 1
                
                # Combine pair coverage with coloring-based quality
                quality_bonus = triple_quality[idx] * 0.5 if coloring is not None else 0
                score = new_pairs + quality_bonus
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1 or best_score <= 0:
                break
            
            # Add the best triple
            active_mask |= (1 << best_idx)
            for pair in triple_pairs[best_idx]:
                covered_pairs.add(pair)
            
            # Remove this triple and its conflicts from remaining
            conflicts = conflict_bits[best_idx]
            remaining_triples = [i for i in remaining_triples if i != best_idx and not ((conflicts >> i) & 1)]
        
        # Greedy refill: add any remaining non-conflicting triples
        for idx in range(num_triples):
            if (active_mask >> idx) & 1:
                continue
            if (conflict_bits[idx] & active_mask) == 0:
                active_mask |= (1 << idx)
        
        return active_mask
    
    def local_search(mask, max_iterations=100):
        """Local search to minimize independence number."""
        current_alpha = find_max_independent_set(mask)
        current_mask = mask
        
        for it in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Strategy: try swapping - remove one active triple, add one inactive triple
            active_indices = [i for i in range(num_triples) if (current_mask >> i) & 1]
            inactive_indices = [i for i in range(num_triples) if not ((current_mask >> i) & 1)]
            
            if not active_indices or not inactive_indices:
                break
            
            # Try a few random swaps
            improved = False
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                
                rem_idx = random.choice(active_indices)
                add_idx = random.choice(inactive_indices)
                
                # Check if adding add_idx is possible after removing rem_idx
                new_mask = current_mask & ~(1 << rem_idx)
                
                # Remove conflicts with add_idx
                conflicts = conflict_bits[add_idx] & new_mask
                new_mask = new_mask & ~conflicts
                new_mask |= (1 << add_idx)
                
                # Quick greedy refill
                for idx in range(num_triples):
                    if (new_mask >> idx) & 1:
                        continue
                    if (conflict_bits[idx] & new_mask) == 0:
                        new_mask |= (1 << idx)
                
                new_alpha = find_max_independent_set(new_mask)
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    current_mask = new_mask
                    improved = True
                    break
                elif new_alpha == current_alpha and random.random() < 0.1:
                    # Accept with small probability for exploration
                    current_mask = new_mask
            
        return current_mask, current_alpha
    
    best_alpha = 16
    best_mask = 0
    
    # Phase 1: Use simulated annealing to find good colorings
    # Phase 2: Use those colorings to guide greedy construction
    # Phase 3: Local search
    
    # Try several different colorings from simulated annealing
    num_colorings = 5
    colorings = []
    for _ in range(num_colorings):
        if time.time() > deadline - 40:
            break
        coloring, score = simulate_annealing_coloring(num_iterations=300)
        colorings.append(coloring)
    
    # If no colorings were generated (time pressure), use a default balanced coloring
    if not colorings:
        colorings = [[0]*8 + [1]*8]
    
    # Multiple restarts with coloring-guided greedy initial construction
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Pick a coloring (cycle through them)
        coloring = colorings[restart % len(colorings)]
        
        # Generate initial solution using greedy pair-coverage with coloring bias
        current_mask = generate_greedy_initial(coloring)
        current_mask, current_alpha = local_search(current_mask, max_iterations=80)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_mask = current_mask
        
        # If we've reached a very good solution, do more polishing
        if best_alpha <= 6 and time.time() < deadline - 20:
            best_mask, best_alpha = local_search(best_mask, max_iterations=100)
    
    # Final polish
    if time.time() < deadline - 5:
        best_mask, best_alpha = local_search(best_mask, max_iterations=50)
    
    # Convert mask to list of triples
    result = []
    for i in range(num_triples):
        if (best_mask >> i) & 1:
            result.append(list(all_triples_list[i]))
    
    return result
