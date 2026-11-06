"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 20
    m = 10
    K = 20  # number of available colours
    
    left = list(range(m))
    right = list(range(m, 2*m))
    
    # Precompute all 4-cycles in K_{m,m}
    # A 4-cycle is (l1, l2, r1, r2) with l1!=l2, r1!=r2
    # Cycle: l1-r1-l2-r2-l1
    cycles = []
    for l1 in range(m):
        for l2 in range(l1+1, m):
            for r1 in range(m):
                for r2 in range(r1+1, m):
                    cycles.append((l1, l2, r1, r2))
    
    def check_colouring(colour_matrix):
        """Check if all 4-cycles are rainbow. colour_matrix[l][r] = color."""
        for l1, l2, r1, r2 in cycles:
            c1 = colour_matrix[l1][r1]
            c2 = colour_matrix[l2][r1]
            c3 = colour_matrix[l2][r2]
            c4 = colour_matrix[l1][r2]
            if len({c1, c2, c3, c4}) < 4:
                return False
        return True
    
    def count_violations(colour_matrix):
        """Count number of violated 4-cycles."""
        count = 0
        for l1, l2, r1, r2 in cycles:
            c1 = colour_matrix[l1][r1]
            c2 = colour_matrix[l2][r1]
            c3 = colour_matrix[l2][r2]
            c4 = colour_matrix[l1][r2]
            if len({c1, c2, c3, c4}) < 4:
                count += 1
        return count
    
    # Strategy 1: Try structured approaches - these are fast
    # Try C[l][r] = (a*l + b*r) % K for various a, b
    found_valid = False
    best_edges = None
    best_count = 0
    
    for a in range(1, K):
        for b in range(1, K):
            if time.time() > deadline - 5:
                break
            colour_matrix = [[(a*l + b*r) % K for r in range(m)] for l in range(m)]
            if check_colouring(colour_matrix):
                found_valid = True
                best_count = m * m
                best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                             for l in range(m) for r in range(m)]
                break
        if found_valid:
            break
    
    if not found_valid:
        # Try C[l][r] = (l * r + c) % K
        for c in range(K):
            if time.time() > deadline - 5:
                break
            colour_matrix = [[(l * r + c) % K for r in range(m)] for l in range(m)]
            if check_colouring(colour_matrix):
                found_valid = True
                best_count = m * m
                best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                             for l in range(m) for r in range(m)]
                break
    
    if not found_valid:
        # Try C[l][r] = (a*l + b*r + c) % K
        for a in range(1, K):
            for b in range(1, K):
                for c in range(K):
                    if time.time() > deadline - 5:
                        break
                    colour_matrix = [[(a*l + b*r + c) % K for r in range(m)] for l in range(m)]
                    if check_colouring(colour_matrix):
                        found_valid = True
                        best_count = m * m
                        best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                                     for l in range(m) for r in range(m)]
                        break
                if found_valid:
                    break
            if found_valid:
                break
    
    # Strategy 2: Random search with local search / repair
    if not found_valid:
        # Use simulated annealing style approach
        best_matrix = None
        best_violations = float('inf')
        
        for attempt in range(500):
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            colour_matrix = [[random.randint(0, K-1) for _ in range(m)] for _ in range(m)]
            violations = count_violations(colour_matrix)
            
            if violations < best_violations:
                best_violations = violations
                best_matrix = [row[:] for row in colour_matrix]
            
            if violations == 0:
                found_valid = True
                best_count = m * m
                best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                             for l in range(m) for r in range(m)]
                break
            
            # Local search: try to reduce violations
            for repair_iter in range(500):
                if time.time() > deadline - 3:
                    break
                
                # Find a violated cycle
                violated = None
                for l1, l2, r1, r2 in cycles:
                    c1 = colour_matrix[l1][r1]
                    c2 = colour_matrix[l2][r1]
                    c3 = colour_matrix[l2][r2]
                    c4 = colour_matrix[l1][r2]
                    if len({c1, c2, c3, c4}) < 4:
                        violated = (l1, l2, r1, r2)
                        break
                
                if violated is None:
                    # Valid!
                    found_valid = True
                    best_count = m * m
                    best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                                 for l in range(m) for r in range(m)]
                    break
                
                l1, l2, r1, r2 = violated
                # Try changing each of the four edges to a random color
                # and pick the one that reduces violations the most
                positions = [(l1, r1), (l2, r1), (l2, r2), (l1, r2)]
                best_change = None
                best_new_violations = violations
                
                for pos in positions:
                    orig_color = colour_matrix[pos[0]][pos[1]]
                    for new_color in range(K):
                        if new_color == orig_color:
                            continue
                        colour_matrix[pos[0]][pos[1]] = new_color
                        new_violations = count_violations(colour_matrix)
                        if new_violations < best_new_violations:
                            best_new_violations = new_violations
                            best_change = (pos, new_color)
                        colour_matrix[pos[0]][pos[1]] = orig_color
                        if best_new_violations == 0:
                            break
                    if best_new_violations == 0:
                        break
                
                if best_change is None:
                    # No improvement possible with single edge change, try random
                    pos = random.choice(positions)
                    colour_matrix[pos[0]][pos[1]] = random.randint(0, K-1)
                    violations = count_violations(colour_matrix)
                else:
                    pos, new_color = best_change
                    colour_matrix[pos[0]][pos[1]] = new_color
                    violations = best_new_violations
                
                if violations < best_violations:
                    best_violations = violations
                    best_matrix = [row[:] for row in colour_matrix]
                
                if violations == 0:
                    found_valid = True
                    best_count = m * m
                    best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                                 for l in range(m) for r in range(m)]
                    break
            
            if found_valid:
                break
        
        # If we found a best_matrix with fewer violations but not zero, 
        # try to continue from there
        if not found_valid and best_matrix is not None and best_violations < float('inf'):
            # Continue local search from best found state
            colour_matrix = [row[:] for row in best_matrix]
            violations = best_violations
            
            for repair_iter in range(2000):
                if time.time() > deadline - 2:
                    break
                
                # Find a violated cycle
                violated = None
                for l1, l2, r1, r2 in cycles:
                    c1 = colour_matrix[l1][r1]
                    c2 = colour_matrix[l2][r1]
                    c3 = colour_matrix[l2][r2]
                    c4 = colour_matrix[l1][r2]
                    if len({c1, c2, c3, c4}) < 4:
                        violated = (l1, l2, r1, r2)
                        break
                
                if violated is None:
                    # Valid!
                    found_valid = True
                    best_count = m * m
                    best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                                 for l in range(m) for r in range(m)]
                    break
                
                l1, l2, r1, r2 = violated
                positions = [(l1, r1), (l2, r1), (l2, r2), (l1, r2)]
                best_change = None
                best_new_violations = violations
                
                for pos in positions:
                    orig_color = colour_matrix[pos[0]][pos[1]]
                    for new_color in range(K):
                        if new_color == orig_color:
                            continue
                        colour_matrix[pos[0]][pos[1]] = new_color
                        new_violations = count_violations(colour_matrix)
                        if new_violations < best_new_violations:
                            best_new_violations = new_violations
                            best_change = (pos, new_color)
                        colour_matrix[pos[0]][pos[1]] = orig_color
                        if best_new_violations == 0:
                            break
                    if best_new_violations == 0:
                        break
                
                if best_change is None:
                    pos = random.choice(positions)
                    colour_matrix[pos[0]][pos[1]] = random.randint(0, K-1)
                    violations = count_violations(colour_matrix)
                else:
                    pos, new_color = best_change
                    colour_matrix[pos[0]][pos[1]] = new_color
                    violations = best_new_violations
                
                if violations == 0:
                    found_valid = True
                    best_count = m * m
                    best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                                 for l in range(m) for r in range(m)]
                    break
    
    # Strategy 3: If K_{10,10} coloring not found, use C4-free greedy bipartite
    if best_edges is None:
        # Build C4-free bipartite graph greedily
        adjL = [set() for _ in range(m)]
        adjR = [set() for _ in range(m)]
        
        edges = []
        all_edges = [(i, j) for i in range(m) for j in range(m)]
        random.shuffle(all_edges)
        
        for i, j in all_edges:
            if time.time() > deadline - 2:
                break
            
            # Check if adding (i, j) creates a C4
            creates_c4 = False
            for i_prime in adjR[j]:
                if adjL[i] & adjL[i_prime]:
                    creates_c4 = True
                    break
            
            if not creates_c4:
                adjL[i].add(j)
                adjR[j].add(i)
                c = random.randint(0, K - 1)
                edges.append([left[i], right[j], c])
        
        best_edges = edges
        best_count = len(edges)
    
    # Strategy 4: Try to add more edges to a valid K_{10,10} coloring
    # If we have a valid K_{10,10}, try adding edges within left or within right
    if found_valid and best_edges is not None:
        # Build adjacency from best_edges
        adj = [[False] * n for _ in range(n)]
        for u, v, c in best_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Try adding edges within left partition or within right partition
        # These edges don't create 4-cycles with the bipartite edges
        # (since 4-cycles in a bipartite graph must alternate between partitions)
        # But wait - adding edges within a partition could create 4-cycles
        # that include edges from both the within-partition and bipartite parts
        
        # Actually, let's be careful. A 4-cycle could be:
        # l1 - l2 - r1 - r2 - l1 if we have edges l1-l2, l2-r1, r1-r2, r2-l1
        # But r1-r2 is not in our bipartite graph (both in right partition)
        # So we'd need to add r1-r2 too.
        
        # Let's just try adding all possible within-partition edges and check
        additional_edges = []
        
        # Try left-left edges
        for i in range(m):
            for j in range(i+1, m):
                if time.time() > deadline - 2:
                    break
                u, v = left[i], left[j]
                if not adj[u][v]:
                    # Check if adding this edge creates a non-rainbow 4-cycle
                    # A 4-cycle involving u-v would need: u-v-x-y-u
                    # where x and y are other vertices
                    # Since u,v are both in left, x,y must be in right
                    # (because edges from left only go to right in our bipartite part)
                    # So the cycle would be: u-v-x-y-u
                    # Edges: u-v (new), v-x, x-y, y-u
                    # But x-y would need to be an edge between two right vertices,
                    # which we don't have unless we add it
                    # So actually, adding a single within-partition edge
                    # cannot create a 4-cycle by itself
                    # It would need another within-partition edge
                    # So it's safe to add!
                    c = random.randint(0, K - 1)
                    additional_edges.append([u, v, c])
                    adj[u][v] = True
                    adj[v][u] = True
        
        # Try right-right edges
        for i in range(m):
            for j in range(i+1, m):
                if time.time() > deadline - 2:
                    break
                u, v = right[i], right[j]
                if not adj[u][v]:
                    c = random.randint(0, K - 1)
                    additional_edges.append([u, v, c])
                    adj[u][v] = True
                    adj[v][u] = True
        
        # Now check if the full graph is still valid
        # We need to check all 4-cycles in the full graph
        # This is more complex, so let's just verify
        all_edges_full = best_edges + additional_edges
        
        # Build edge color map for quick lookup
        edge_color = {}
        for u, v, c in all_edges_full:
            edge_color[(min(u,v), max(u,v))] = c
            edge_color[(max(u,v), min(u,v))] = c
        
        # Check all 4-cycles
        valid = True
        vertices = list(range(n))
        # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges a-b, b-c, c-d, d-a
        # We need to check all such cycles
        # For efficiency, only check cycles where the edges exist
        for a in range(n):
            for b in range(a+1, n):
                if (a,b) not in edge_color:
                    continue
                for c in range(b+1, n):
                    if (b,c) not in edge_color:
                        continue
                    for d in range(c+1, n):
                        if (c,d) not in edge_color or (d,a) not in edge_color:
                            continue
                        # Check if this is a valid 4-cycle (no diagonals that would make it not a cycle)
                        # Actually, for a 4-cycle we just need the 4 edges to exist
                        # and the 4 vertices to be distinct
                        c1 = edge_color[(a,b)]
                        c2 = edge_color[(b,c)]
                        c3 = edge_color[(c,d)]
                        c4 = edge_color[(d,a)]
                        if len({c1, c2, c3, c4}) < 4:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_edges = all_edges_full
            best_count = len(best_edges)
        else:
            # The additional edges broke validity, so just use the original
            pass
    
    return best_edges
# EVOLVE-BLOCK-END
