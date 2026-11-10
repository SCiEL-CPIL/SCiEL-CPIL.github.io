import itertools
import random

def construct():
       # Generate STS(15)
       # Points are non-zero 4-bit vectors
       points = []
       for i in range(1, 16):
           points.append(i) # 1..15 represent binary 0001..1111
       
       # Map point value to index 0..14
       # Let's just use the values 1..15 as indices 0..14?
       # Better: index = value - 1
       
       edges = []
       # Generate all lines
       # A line is {u, v, u^v} for u != v, u^v != 0
       # To avoid duplicates, we can iterate u < v < u^v?
       # Or just add set and deduplicate.
       edge_set = set()
       for i in range(15):
           u = i + 1
           for j in range(i+1, 15):
               v = j + 1
               w = u ^ v
               if w != 0:
                   # Sort the triple to have a canonical representation
                   triple = tuple(sorted([u, v, w]))
                   edge_set.add(triple)
       
       # Convert to list of lists
       sts15_edges = [list(t) for t in edge_set]
       
       # Map values 1..15 to indices 0..14
       # The points in the hypergraph will be 0..15
       # STS points are 0..14. New point is 15.
       # So we subtract 1 from all elements in sts15_edges.
       sts15_edges = [[x-1 for x in e] for e in sts15_edges]
       
       best_alpha = 17
       best_edges = []
       
       # Try different matchings
       # We'll try a few random matchings of size 7, 6, 5
       # And also the empty matching (k=0)
       
       def get_alpha(edges):
           # edges is list of lists of 3 ints (0..15)
           # Check all subsets of size k from 16 down to 1
           # To speed up, we can use bitmasks
           n = 16
           # Precompute bitmask for each edge
           edge_masks = [0] * len(edges)
           for i, e in enumerate(edges):
               mask = (1 << e[0]) | (1 << e[1]) | (1 << e[2])
               edge_masks[i] = mask
           
           # We want to find max k such that there exists a mask S of weight k
           # such that for all i, (S & edge_masks[i]) != edge_masks[i]
           # i.e., no edge is fully contained in S.
           
           # Iterate k from 16 down to 1
           for k in range(16, 0, -1):
               # Generate all combinations of k vertices
               # Using itertools.combinations
               found = False
               for comb in itertools.combinations(range(n), k):
                   mask = 0
                   for v in comb:
                       mask |= (1 << v)
                   
                   # Check if any edge is contained in mask
                   is_independent = True
                   for m in edge_masks:
                       if (mask & m) == m:
                           is_independent = False
                           break
                   
                   if is_independent:
                       found = True
                       break
               
               if found:
                   return k
           return 0

       # Generate candidates
       candidates = []
       
       # Candidate 1: No modification
       candidates.append(sts15_edges)
       
       # Candidate 2..N: Random matchings
       vertices_15 = list(range(15))
       for _ in range(50): # 50 random trials
           random.shuffle(vertices_15)
           # Pick 7 disjoint pairs
           # Since we shuffled, we can just take consecutive pairs?
           # No, that's not a matching in the sense of disjointness if we just take pairs from a shuffled list?
           # Yes, if we take (v0,v1), (v2,v3)... they are disjoint.
           matching = []
           for i in range(0, 14, 2):
               matching.append((vertices_15[i], vertices_15[i+1]))
           
           # Construct new edges
           new_edges = []
           # Build a set of edges from STS15 for fast lookup?
           # We need to remove edges containing the pairs in matching.
           # And add new edges [15, u, v].
           
           # Map pair to edge in STS15
           # Since STS15 is a Steiner system, each pair is in exactly one edge.
           # We can precompute a dictionary: pair -> edge_index
           pair_to_edge = {}
           for idx, e in enumerate(sts15_edges):
               pairs = [(e[0], e[1]), (e[1], e[2]), (e[0], e[2])]
               for p in pairs:
                   key = (min(p), max(p))
                   pair_to_edge[key] = idx
           
           edges_to_remove = set()
           for u, v in matching:
               key = (min(u, v), max(u, v))
               if key in pair_to_edge:
                   edges_to_remove.add(pair_to_edge[key])
           
           # Create new edge list
           for idx, e in enumerate(sts15_edges):
               if idx not in edges_to_remove:
                   new_edges.append(e)
           
           for u, v in matching:
               new_edges.append([15, u, v])
           
           candidates.append(new_edges)
       
       # Evaluate all candidates
       for edges in candidates:
           # Verify linearity just in case
           valid = True
           pair_count = {}
           for e in edges:
               for i in range(3):
                   for j in range(i+1, 3):
                       p = (min(e[i], e[j]), max(e[i], e[j]))
                       pair_count[p] = pair_count.get(p, 0) + 1
                       if pair_count[p] > 1:
                           valid = False
                           break
                   if not valid: break
               if not valid: break
           
           if not valid:
               continue
               
           alpha = get_alpha(edges)
           if alpha < best_alpha:
               best_alpha = alpha
               best_edges = edges
       
       return best_edges
