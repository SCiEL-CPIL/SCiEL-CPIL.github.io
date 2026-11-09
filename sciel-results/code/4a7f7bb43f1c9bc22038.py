"""
Strategy: Simulated Annealing on Linear Triple Systems
We start with a valid linear triple system (e.g., from a known construction or greedy).
We then use Simulated Annealing to minimize the independence number.
The state is a list of triples. Moves are swapping two triples.
The cost function is the independence number of the hypergraph.
To make the cost function fast, we use bitmasks for subsets and precomputed triple masks.
"""

import time
import random
import sys

# Constants
N_POINTS = 16
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    
    # 1. Generate initial valid linear triple system
    # We need a set of triples on 16 points such that any two share at most 1 point.
    # A simple way is to take a Steiner Triple System (STS) if available, but STS(16) doesn't exist.
    # We can construct one by hand or use a greedy method to fill as many triples as possible
    # while maintaining linearity, then pad if necessary (though we don't need to maximize triples,
    # just minimize alpha). Actually, having more triples generally helps reduce alpha,
    # as long as linearity is maintained.
    
    # Let's try to construct a dense linear triple system.
    # Max number of triples in a linear triple system on v points is bounded by v(v-1)/6.
    # For v=16, 16*15/6 = 40.
    # We can use a greedy approach to find a large set of disjoint-in-pairs triples.
    
    all_triples = []
    for i in range(N_POINTS):
        for j in range(i+1, N_POINTS):
            for k in range(j+1, N_POINTS):
                all_triples.append((i, j, k))
    
    # Greedy construction of a linear triple system
    # We want to select a subset T of all_triples such that any two triples in T share at most 1 point.
    # This is equivalent to: for any two triples (a,b,c) and (d,e,f) in T, they don't share 2 or 3 points.
    # i.e., they don't contain the same pair.
    
    # We can model this as finding a large matching in a 3-uniform hypergraph where edges are pairs?
    # No, it's simpler: We are selecting triples such that no pair of points is contained in more than one triple.
    # This is a Partial Steiner Triple System.
    
    # Let's build it greedily.
    selected_triples = []
    used_pairs = set()
    
    # Shuffle to allow different starting points, but deterministic is fine for reproducibility if we seed random
    random.seed(42)
    shuffled_triples = all_triples[:]
    random.shuffle(shuffled_triples)
    
    for t in shuffled_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        if all(p not in used_pairs for p in pairs):
            selected_triples.append(t)
            for p in pairs:
                used_pairs.add(p)
        
        # Early stop if we have enough? 
        # Max possible is 40. Let's see how many we get.
        # If we run out of candidates, we stop.
        
    # If the greedy method didn't find many, we might have a sparse system.
    # A sparse system has high alpha. We want to maximize the number of triples to minimize alpha.
    # Let's check the size.
    # If size is small, we can try multiple random restarts of the greedy phase.
    
    best_initial_system = []
    best_size = 0
    
    for _ in range(20): # 20 restarts for initial construction
        random.shuffle(shuffled_triples)
        selected = []
        used = set()
        for t in shuffled_triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            if all(p not in used for p in pairs):
                selected.append(t)
                for p in pairs:
                    used.add(p)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_initial_system = selected[:]
            
    current_system = best_initial_system[:]
    
    # If we didn't find any (unlikely), fallback to disjoint triples
    if not current_system:
        current_system = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
        
    # Precompute masks for the current system's triples for fast independence checking
    # Actually, for the SA loop, we need to evaluate alpha for many configurations.
    # Computing alpha from scratch is O(2^N * |T|) which is 2^16 * 40 ~ 2.6M ops.
    # 2.6M ops is very fast in Python if optimized, maybe 0.1s?
    # 145s / 0.1s = 1450 iterations. That's not a lot for SA.
    # We need a faster way to estimate or compute alpha, or accept fewer iterations.
    
    # Alternative: Use a greedy heuristic to find a large independent set, which gives a lower bound on alpha.
    # If we find an independent set of size K, alpha >= K.
    # We want to MINIMIZE alpha. So we want to destroy large independent sets.
    # Cost function: Size of the largest independent set found by a fast heuristic (e.g., greedy coloring complement).
    # If the heuristic says "found independent set of size 8", cost = 8.
    # This is much faster than exact alpha computation.
    
    def heuristic_alpha(system):
        """
        Estimates the independence number by finding a large independent set greedily.
        An independent set is a set of points containing no triple.
        This is equivalent to finding a large set of points such that no triple is fully contained.
        
        We can view this as:
        Start with all points.
        While there is a triple fully contained in the current set:
            Remove one point from that triple (the one that breaks the most other triples? or random?)
        
        This finds a "2-packing" or similar? No.
        
        Actually, let's use a different heuristic.
        We want to find S such that no triple is in S.
        This is the complement of a vertex cover of the hypergraph? No.
        
        Let's just use a simple local search for an independent set.
        Start with S = all points.
        While exists triple t subset S:
            Remove a point p from t that is in the most triples within S.
        
        This will converge to a maximal independent set (not necessarily maximum, but large).
        """
        S_mask = (1 << N_POINTS) - 1
        # Precompute masks for triples in system
        triple_masks = [ (1<<t[0]) | (1<<t[1]) | (1<<t[2]) for t in system ]
        
        # Count how many triples contain each point, within S
        # We update this dynamically or just recalculate?
        # For speed, let's just pick a random triple and remove a random point from it.
        # Repeat until no triple is in S.
        
        # To be faster, we can check if any triple is in S by checking if (t_mask & S_mask) == t_mask.
        # There are at most 40 triples. 40 checks is fast.
        
        changed = True
        while changed:
            changed = False
            for t_mask in triple_masks:
                if (t_mask & S_mask) == t_mask:
                    # This triple is in S. We must remove at least one point.
                    # Pick the point in t_mask that is in S.
                    # Let's pick the one that appears in the most other triples in S.
                    points_in_t = [i for i in range(N_POINTS) if (t_mask >> i) & 1]
                    # Heuristic: remove the point that is in the most triples
                    counts = [0]*N_POINTS
                    for other_mask in triple_masks:
                        if (other_mask & S_mask) == other_mask:
                            for p in points_in_t:
                                if (other_mask >> p) & 1:
                                    counts[p] += 1
                    # Find max count
                    max_count = max(counts)
                    # Pick first point with max count
                    for p in points_in_t:
                        if counts[p] == max_count:
                            S_mask &= ~(1 << p)
                            changed = True
                            break
                    if changed:
                        break # Restart loop as S changed
        return bin(S_mask).count('1')

    # Initial cost
    current_cost = heuristic_alpha(current_system)
    best_system = current_system[:]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    while time.time() - start_time < DEADLINE:
        if T < min_T:
            break
            
        # Generate a move: swap two triples
        # We can also try adding a triple or removing one?
        # Adding a triple is hard because we need to maintain linearity.
        # Swapping is safer if we ensure the new triple doesn't violate linearity with others.
        
        # Let's stick to swaps for simplicity, or "kick" moves.
        # A "kick" move: remove a triple, try to add a new one that fits.
        
        # Let's try: Pick a random triple in the system, remove it.
        # Then try to add a random triple from all_triples that is compatible with the remaining system.
        
        if not current_system:
            break
            
        idx1 = random.randint(0, len(current_system) - 1)
        old_triple = current_system[idx1]
        
        # Remove old_triple
        temp_system = current_system[:]
        temp_system.pop(idx1)
        
        # Find a new triple that is compatible with temp_system
        # Compatibility: shares at most 1 point with every triple in temp_system.
        # This is equivalent to: no pair in new_triple is in any triple in temp_system.
        
        # Let's precompute pairs used in temp_system for this step? 
        # That's O(N^2) per move. Too slow?
        # 40 triples * 3 pairs = 120 pairs.
        # Checking a candidate triple against 120 pairs is fast.
        
        # Let's pick a random candidate triple
        found_new = False
        new_triple = None
        attempts = 0
        while attempts < 50: # Limit attempts
            candidate = random.choice(all_triples)
            # Check compatibility
            c_pairs = [(candidate[0], candidate[1]), (candidate[0], candidate[2]), (candidate[1], candidate[2])]
            valid = True
            # Check against all triples in temp_system
            for t in temp_system:
                t_pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
                # Check intersection of pairs
                # If any pair is common, invalid.
                for cp in c_pairs:
                    if cp in t_pairs:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_triple = candidate
                found_new = True
                break
            attempts += 1
            
        if found_new:
            # Perform move
            current_system[idx1] = new_triple
            new_cost = heuristic_alpha(current_system)
            
            # Accept if better or with probability exp(-(delta)/T)
            delta = new_cost - current_cost
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_system = current_system[:]
            else:
                # Accept with probability
                # delta is integer >= 0
                # prob = exp(-delta / T)
                if delta == 0:
                    current_cost = new_cost
                else:
                    # Use random.random() < exp(-delta/T)
                    # To avoid underflow, if delta/T is large, prob is 0
                    if delta < 100 * T: # Rough cutoff
                        if random.random() < pow(2.718281828, -delta / T):
                            current_cost = new_cost
        else:
            # If we couldn't find a compatible triple, maybe we just keep the old one?
            # Or we could try to remove the triple entirely (shrink system)?
            # Shrinking usually increases alpha, so bad.
            # Let's just do nothing (stay) or try a different move.
            pass
            
        T *= cooling_rate
        
    return best_system

# Fallback in case of any error
if __name__ == "__main__":
    # Simple test
    sys = construct()
    print(f"System size: {len(sys)}")
    # Verify linearity
    for i in range(len(sys)):
        for j in range(i+1, len(sys)):
            common = set(sys[i]) & set(sys[j])
            if len(common) > 1:
                print("Invalid system!")
                break
