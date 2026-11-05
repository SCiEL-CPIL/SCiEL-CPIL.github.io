import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n_points = 16
    points = list(range(n_points))
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        triple_pairs.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
    
    # Map pair to list of triple indices containing it
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)
            
    # We want to build a linear triple system (LTS) that minimizes the independence number.
    # An independent set is a set of points containing no triple.
    # Equivalently, we want to "hit" all large subsets of points.
    
    # Strategy: Greedily add triples.
    # At each step, choose the triple that, when added to the current LTS,
    # most reduces the size of the largest independent set.
    
    # To estimate the independence number efficiently:
    # We can maintain a "conflict graph" where nodes are points, and an edge exists
    # if the two points are in a selected triple? No, that's not quite right.
    # An independent set S contains no triple. 
    # If we select a set of triples T, S is independent if for all t in T, t is not a subset of S.
    # This is equivalent to saying S is an independent set in the hypergraph (V, T).
    
    # A simple heuristic for alpha(T):
    # Start with all points.
    # Repeatedly remove points that are part of a "forced" structure?
    # Or, use a greedy algorithm to find a large independent set:
    # 1. Start with V = {0..15}
    # 2. While there is a triple t in T such that t subset of V:
    #    Remove one point from t from V.
    #    (Which point? Random or the one with highest degree in T?)
    # The size of V at the end is a lower bound on alpha.
    # To get a better estimate, we can run this multiple times with different removal strategies.
    
    def estimate_alpha(selected_triples, num_trials=5):
        # selected_triples is a list of tuple indices
        if not selected_triples:
            return n_points
            
        # Build adjacency or incidence structure for quick lookup
        # For each point, list of triples in selected_triples containing it
        point_triples = {p: [] for p in points}
        for idx in selected_triples:
            t = all_triples[idx]
            for p in t:
                point_triples[p].append(idx)
                
        best_lower_bound = 0
        for _ in range(num_trials):
            V = set(points)
            # Greedy removal
            # We want to remove points to "break" all triples.
            # This is the Hitting Set problem. The complement of a hitting set is an independent set.
            # Min Hitting Set -> Max Independent Set.
            # Greedy Hitting Set:
            # While there is an uncovered triple:
            #   Pick the point that covers the most uncovered triples.
            #   Remove it from V (add to Hitting Set).
            
            uncovered_triples = set(selected_triples)
            if not uncovered_triples:
                return n_points # Empty LTS, alpha = 16
            
            # Count how many uncovered triples each point covers
            point_counts = {p: 0 for p in points}
            for idx in uncovered_triples:
                t = all_triples[idx]
                for p in t:
                    point_counts[p] += 1
                    
            while uncovered_triples:
                # Find point with max count
                max_count = 0
                best_point = None
                for p in points:
                    if point_counts[p] > max_count:
                        max_count = point_counts[p]
                        best_point = p
                        
                if best_point is None or max_count == 0:
                    break
                    
                # Remove best_point from V
                V.remove(best_point)
                
                # Update uncovered_triples and point_counts
                # All triples containing best_point are now hit (covered)
                # Remove them from uncovered_triples
                new_uncovered = set()
                for idx in uncovered_triples:
                    if best_point in all_triples[idx]:
                        # This triple is hit
                        # Reduce counts for other points in this triple
                        t = all_triples[idx]
                        for p in t:
                            if p != best_point:
                                point_counts[p] -= 1
                    else:
                        new_uncovered.add(idx)
                
                uncovered_triples = new_uncovered
                
            # V is now a candidate independent set?
            # Wait, the greedy hitting set removes points to hit all triples.
            # The remaining points V do NOT necessarily form an independent set.
            # Actually, if we remove points to hit all triples, the remaining set V contains NO triple.
            # So V is an independent set.
            # Its size is a lower bound on alpha.
            if len(V) > best_lower_bound:
                best_lower_bound = len(V)
                
        return best_lower_bound

    # We want to MINIMIZE alpha.
    # So we want to ADD triples that REDUCE the estimated alpha the most.
    
    # Initial state: no triples. alpha = 16.
    # If we add a triple, we can potentially reduce alpha.
    
    # Let's try to build a system with a target number of triples.
    # A Steiner Triple System on 15 points has 35 triples.
    # On 16 points, a maximal linear triple system might have more.
    # Let's aim for a dense system.
    
    # Greedy Construction:
    selected = []
    used_pairs = set()
    
    # Pre-sort triples?
    # We can try multiple random orders or a deterministic score.
    
    # To make it deterministic and fast, we will use a scoring function.
    # Score of a triple t = reduction in alpha if t is added.
    # Calculating this exactly is expensive.
    # Approximation:
    # Adding a triple t forbids t as a subset of any independent set.
    # It effectively "removes" the possibility of having all 3 points of t.
    
    # Let's use a different heuristic:
    # We want to cover pairs. A linear triple system is a packing.
    # The more pairs we cover, the "denser" the system, and typically the smaller the independence number.
    # So, greedily add triples that cover the most "new" pairs?
    # All triples cover 3 new pairs if they are valid (linear).
    # So we just need to find a maximal packing.
    # But not all packings have the same independence number.
    
    # Let's try to find a packing that is "well-distributed".
    # We can use a greedy algorithm that picks the triple with the lowest "conflict" with existing triples,
    # or simply the first valid one in a shuffled list.
    
    # Let's try a few random packings and evaluate their independence number using a better estimator.
    
    def evaluate_packing(packing):
        # Use the hitting set heuristic to estimate alpha
        # We can run it with more trials for accuracy
        return estimate_alpha(packing, num_trials=20)
        
    best_packing = None
    best_alpha = 16
    
    # Time limit for search
    search_deadline = time.time() + 100.0
    
    # Try multiple random restarts
    for restart in range(10000):
        if time.time() > search_deadline:
            break
            
        random.shuffle(all_triples)
        
        current_packing = []
        current_used_pairs = set()
        
        for i, t in enumerate(all_triples):
            pairs = triple_pairs[i]
            valid = True
            for p in pairs:
                if p in current_used_pairs:
                    valid = False
                    break
            if valid:
                current_packing.append(i)
                for p in pairs:
                    current_used_pairs.add(p)
                    
        # Evaluate this packing
        # To speed up evaluation, we can stop early if alpha is already >= best_alpha
        # But estimate_alpha is a lower bound. If lower bound >= best_alpha, we can't conclude it's worse.
        # We need a better evaluation or more trials.
        
        # Let's use a quick check:
        # If the packing is maximal (no more triples can be added), it's a candidate.
        # Check if maximal:
        is_maximal = True
        for i, t in enumerate(all_triples):
            if i in current_packing:
                continue
            pairs = triple_pairs[i]
            valid = True
            for p in pairs:
                if p in current_used_pairs:
                    valid = False
                    break
            if valid:
                is_maximal = False
                break
                
        if not is_maximal:
            continue
            
        # Evaluate alpha
        # We can use a simpler upper bound for alpha to prune?
        # No, we want to minimize alpha.
        
        alpha_est = evaluate_packing(current_packing)
        
        if alpha_est < best_alpha:
            best_alpha = alpha_est
            best_packing = current_packing[:]
            
            # If we find alpha <= 8, it's very good.
            # Theoretical minimum for n=16?
            # If alpha=8, we can partition 16 points into 2 sets of 8, each containing no triple.
            # If alpha=7, even better.
            
            if best_alpha <= 7:
                break
                
    if best_packing is None:
        # Fallback: return a simple valid system
        # Just return a few triples
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [0, 3, 6], [1, 4, 7]]
        
    # Convert indices to triples
    result = [list(all_triples[idx]) for idx in best_packing]
    
    return result

# Test locally
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Num triples: {len(res)}")
    
    # Verify linearity
    pairs = set()
    for t in res:
        p1, p2, p3 = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
        if p1 in pairs or p2 in pairs or p3 in pairs:
            print("ERROR: Not linear!")
            break
        pairs.add(p1)
        pairs.add(p2)
        pairs.add(p3)
        
    # Estimate alpha
    print(f"Estimated Alpha: {estimate_alpha([all_triples.index(tuple(t)) for t in res], 20)}")
