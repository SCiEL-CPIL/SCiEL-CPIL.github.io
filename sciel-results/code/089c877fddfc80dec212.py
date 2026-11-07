"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 70
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    # Greedy maximal biclique finding
    def find_maximal_biclique(uncovered_edges, adj):
        """Find a large biclique from uncovered edges."""
        if not uncovered_edges:
            return None, None, set()
        
        # Pick the edge with highest degree in uncovered
        best_edge = None
        best_score = -1
        for e in uncovered_edges:
            u, v = e
            # Score: how many other uncovered edges share u or v
            score = 0
            for e2 in uncovered_edges:
                if e2[0] == u or e2[0] == v or e2[1] == u or e2[1] == v:
                    score += 1
            if score > best_score:
                best_score = score
                best_edge = e
        
        u, v = best_edge
        left = {u}
        right = {v}
        
        # Expand greedily
        changed = True
        while changed:
            changed = False
            # Try adding to left
            for w in adj[u]:
                if w not in left and w != v:
                    # w can join left if w is adjacent to all in right
                    if all(w in adj[r] for r in right):
                        # Check if (w, r) is an edge for all r in right
                        ok = True
                        for r in right:
                            if (min(w,r), max(w,r)) not in edge_set:
                                ok = False
                                break
                        if ok:
                            left.add(w)
                            changed = True
            # Try adding to right
            for w in adj[v]:
                if w not in right and w != u:
                    if all(w in adj[l] for l in left):
                        ok = True
                        for l in left:
                            if (min(w,l), max(w,l)) not in edge_set:
                                ok = False
                                break
                        if ok:
                            right.add(w)
                            changed = True
        
        # Compute covered edges
        covered = set()
        for l in left:
            for r in right:
                e = (min(l,r), max(l,r))
                if e in uncovered_edges:
                    covered.add(e)
        
        if not covered:
            return None, None, set()
        
        return list(left), list(right), covered
    
    # Greedy partition
    uncovered = set(edge_set)
    bicliques = []
    
    while uncovered and time.time() < deadline:
        left, right, covered = find_maximal_biclique(uncovered, adj)
        if left is None:
            # Fallback: single edge
            e = next(iter(uncovered))
            u, v = e
            left = [u]
            right = [v]
            covered = {e}
        
        bicliques.append([left, right])
        uncovered -= covered
    
    # Local search: try to merge bicliques
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            for j in range(i+1, len(bicliques)):
                if time.time() > deadline:
                    break
                
                l1, r1 = bicliques[i]
                l2, r2 = bicliques[j]
                
                # Try merging: new_left = l1 ∪ l2, new_right = r1 ∪ r2
                # Check if all cross pairs are edges
                new_left = set(l1) | set(l2)
                new_right = set(r1) | set(r2)
                
                # Verify biclique property
                valid = True
                for a in new_left:
                    for b in new_right:
                        e = (min(a,b), max(a,b))
                        if e not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check that the union of covered edges equals the combined covered edges
                # Compute what edges the merged biclique covers
                merged_covered = set()
                for a in new_left:
                    for b in new_right:
                        e = (min(a,b), max(a,b))
                        if e in edge_set:
                            merged_covered.add(e)
                
                # What was covered by i and j
                old_covered_i = set()
                for a in l1:
                    for b in r1:
                        e = (min(a,b), max(a,b))
                        if e in edge_set:
                            old_covered_i.add(e)
                
                old_covered_j = set()
                for a in l2:
                    for b in r2:
                        e = (min(a,b), max(a,b))
                        if e in edge_set:
                            old_covered_j.add(e)
                
                # The merged biclique must cover exactly old_covered_i ∪ old_covered_j
                # No new edges, no missing edges
                if merged_covered == (old_covered_i | old_covered_j):
                    # Perform merge
                    bicliques[i] = [list(new_left), list(new_right)]
                    bicliques.pop(j)
                    improved = True
                    break
            if improved:
                break
    
    # Try splitting bicliques to allow further merges
    if time.time() < deadline:
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            l, r = bicliques[i]
            if len(l) <= 1 or len(r) <= 1:
                continue
            
            # Try splitting left side into two parts
            for subset_size in range(1, len(l)//2 + 1):
                for subset in itertools.combinations(l, subset_size):
                    if time.time() > deadline:
                        break
                    subset_set = set(subset)
                    left_part1 = list(subset_set)
                    left_part2 = [x for x in l if x not in subset_set]
                    
                    # Both parts must form valid bicliques with r
                    # Part 1: left_part1 x r must be all edges
                    valid1 = True
                    for a in left_part1:
                        for b in r:
                            e = (min(a,b), max(a,b))
                            if e not in edge_set:
                                valid1 = False
                                break
                        if not valid1:
                            break
                    if not valid1:
                        continue
                    
                    # Part 2: left_part2 x r must be all edges
                    valid2 = True
                    for a in left_part2:
                        for b in r:
                            e = (min(a,b), max(a,b))
                            if e not in edge_set:
                                valid2 = False
                                break
                        if not valid2:
                            break
                    if not valid2:
                        continue
                    
                    # Do the split
                    bicliques[i] = [left_part1, list(r)]
                    bicliques.insert(i+1, [left_part2, list(r)])
                    improved = True
                    break
                if improved:
                    break
            if improved:
                # Retry merging
                improved2 = True
                while improved2 and time.time() < deadline:
                    improved2 = False
                    for i in range(len(bicliques)):
                        for j in range(i+1, len(bicliques)):
                            if time.time() > deadline:
                                break
                            l1, r1 = bicliques[i]
                            l2, r2 = bicliques[j]
                            new_left = set(l1) | set(l2)
                            new_right = set(r1) | set(r2)
                            valid = True
                            for a in new_left:
                                for b in new_right:
                                    e = (min(a,b), max(a,b))
                                    if e not in edge_set:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                continue
                            merged_covered = set()
                            for a in new_left:
                                for b in new_right:
                                    e = (min(a,b), max(a,b))
                                    if e in edge_set:
                                        merged_covered.add(e)
                            old_covered_i = set()
                            for a in l1:
                                for b in r1:
                                    e = (min(a,b), max(a,b))
                                    if e in edge_set:
                                        old_covered_i.add(e)
                            old_covered_j = set()
                            for a in l2:
                                for b in r2:
                                    e = (min(a,b), max(a,b))
                                    if e in edge_set:
                                        old_covered_j.add(e)
                            if merged_covered == (old_covered_i | old_covered_j):
                                bicliques[i] = [list(new_left), list(new_right)]
                                bicliques.pop(j)
                                improved2 = True
                                break
                        if improved2:
                            break
    
    # Ensure return types are correct: lists of ints
    result = []
    for l, r in bicliques:
        result.append([[int(x) for x in l], [int(x) for x in r]])
    return result
# EVOLVE-BLOCK-END
