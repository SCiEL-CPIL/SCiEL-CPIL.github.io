"""Algebraic design: affine lines in GF(2)^4 for linear triple system on 16 points."""
import time
import itertools
import random
from collections import defaultdict

def construct():
    """
    Construct a linear triple system on points 0..15 using affine lines in GF(2)^4.
    
    The 16 points are represented as 4-bit vectors. We use a structured set of triples
    derived from affine geometry to ensure linearity (any two triples share at most one point)
    and minimize the independence number.
    
    Strategy: Use all lines of the form {a, b, a+b} in GF(2)^4 where a, b are linearly independent,
    but select a subset that maintains linearity. Actually, in GF(2)^4, a "line" in the affine
    sense has 2 points, not 3. So we need a different approach.
    
    Better approach: Use a Steiner triple system-like construction. For 16 points, we can use
    a partial Steiner triple system. Let's use a greedy approach with backtracking, but seeded
    with a good initial structure.
    
    Actually, let's use the following known construction:
    - Take the 16 points as elements of GF(2)^4.
    - A triple {a, b, c} is "linear" in GF(2)^4 if a + b + c = 0 (i.e., c = a + b).
    - The set of all such triples forms a 3-uniform hypergraph where any two hyperedges share
      at most one vertex (this is a property of linear spaces over GF(2)).
    - However, not all such triples can be included simultaneously while maintaining the
      pairwise intersection property. We need to select a subset.
    
    Let me reconsider. The condition is that any two triples share at most one point. This is
    the definition of a linear 3-uniform hypergraph. 
    
    For GF(2)^4, consider triples of the form {x, y, x+y} where x, y are non-zero and linearly
    independent. Two such triples {x, y, x+y} and {u, v, u+v} share at most one point unless
    they are identical or share two points. They share two points if, e.g., x = u and y = v, or
    x = u and y = u+v, etc. This happens when the sets {x, y} and {u, v} share an element and
    the sums are consistent.
    
    Actually, let's just use a greedy algorithm with a good initialization and local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                common = set(triples[i]) & set(triples[j])
                if len(common) > 1:
                    return False
        return True
    
    def calc_independence_number(triples):
        """
        Calculate the independence number: the largest subset of points containing no triple.
        For n=16, we can check all subsets of size 8, 9, ..., 16.
        Actually, we want the LARGEST set with NO triple, so we check from large to small.
        """
        # Convert to a set of frozensets for fast lookup
        triple_set = set(frozenset(t) for t in triples)
        
        # Check from size 16 down to 1
        for size in range(16, 0, -1):
            for subset in itertools.combinations(range(16), size):
                subset_set = set(subset)
                # Check if any triple is contained in this subset
                has_triple = False
                for t in triples:
                    if all(x in subset_set for x in t):
                        has_triple = True
                        break
                if not has_triple:
                    return size
        return 1
    
    def can_add(triples, new_triple):
        """Check if adding new_triple maintains linearity."""
        for t in triples:
            if len(set(t) & set(new_triple)) > 1:
                return False
        return True
    
    # Initial construction using a structured approach
    # Use the fact that GF(2)^4 has a natural structure
    # Let's try a greedy approach with multiple restarts
    
    best_triples = None
    best_alpha = 16
    
    # Generate candidate triples: all C(16,3) = 560 possible triples
    all_triples = list(itertools.combinations(range(16), 3))
    
    # Try multiple random orderings
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(all_triples)
        triples = []
        used_points = defaultdict(int)  # Count how many triples each point is in
        
        for t in all_triples:
            if time.time() > deadline:
                break
            if can_add(triples, t):
                # Heuristic: prefer triples that cover underutilized points
                triples.append(list(t))
        
        if len(triples) < 5:
            continue
        
        # Calculate independence number
        alpha = calc_independence_number(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
        
        # Local search: try to swap or add triples
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(triples)):
                if time.time() > deadline:
                    break
                # Try removing triple i and adding a different one
                old_triple = triples[i]
                triples[i] = None
                for j, t in enumerate(all_triples):
                    if j == i or t == old_triple:
                        continue
                    if can_add(triples[:i] + triples[i+1:], t):
                        triples[i] = list(t)
                        new_alpha = calc_independence_number(triples)
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_triples = [list(x) for x in triples]
                            improved = True
                        else:
                            triples[i] = old_triple
                        break
    
    # Return the best result found
    if best_triples is not None:
        return best_triples
    
    # Fallback: return the trivial construction
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
