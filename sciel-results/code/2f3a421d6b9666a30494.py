import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique(adj, n, deadline):
        """
        Find a maximum clique in the graph defined by adj.
        Uses Bron-Kerbosch with pivot.
        Returns a bitmask of the vertices in the clique.
        """
        # Precompute degrees for pivot selection heuristic
        # Actually, for dense graphs, a simple pivot works well.
        # We want to maximize the size of the clique.
        
        best_clique_mask = 0
        best_size = 0
        
        # We can use a recursive approach with pruning.
        # P: candidates, X: excluded (already processed)
        # To find MAX clique, we can just run BK and keep the largest found.
        
        def bronk(P, X):
            nonlocal best_clique_mask, best_size
            
            if time.time() > deadline:
                return
                
            if not P and not X:
                # Found a maximal clique
                size = P.bit_count() if P else 0 # Wait, P is empty here.
                # The clique is actually the set of vertices we added along the path.
                # Standard BK tracks the current clique R.
                # Let's modify BK to track R.
                pass
        
        # Better implementation: Standard BK with R, P, X.
        def bk(R, P, X):
            nonlocal best_clique_mask, best_size
            
            if time.time() > deadline:
                return

            if not P and not X:
                # Maximal clique found
                size = R.bit_count()
                if size > best_size:
                    best_size = size
                    best_clique_mask = R
                return

            # Pivot selection: choose u in P U X with max degree in P
            # If P is empty, pivot is 0
            if P:
                pivot_mask = P | X
                # Find u with max (P & adj[u]).bit_count()
                # Optimization: iterate through bits of pivot_mask
                # For dense graphs, almost any pivot works.
                # Let's pick the first one with high degree.
                max_deg = -1
                u = 0
                tmp = pivot_mask
                while tmp:
                    lsb = tmp & -tmp
                    v = lsb.bit_length() - 1
                    deg = (P & adj[v]).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    tmp ^= lsb
                
                # Candidates are P \ N(u)
                candidates = P & ~adj[u]
            else:
                candidates = P # Empty

            # Recurse on candidates
            # Note: We must iterate over candidates.
            # To prune effectively, we should order them.
            # For now, just iterate.
            tmp_cand = candidates
            while tmp_cand:
                lsb = tmp_cand & -tmp_cand
                v = lsb.bit_length() - 1
                
                # Pruning: if |R| + |P| <= best_size, we can't improve?
                # Actually, standard BK finds ALL maximal cliques.
                # To find ONLY the maximum, we can add a bound:
                # if R.bit_count() + P.bit_count() <= best_size: return
                if R.bit_count() + P.bit_count() <= best_size:
                    return

                bk(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb
                
                tmp_cand ^= lsb
                
                if time.time() > deadline:
                    return

        # Initialize
        # We want to find a large clique.
        # Start with all vertices.
        bk(0, (1 << n) - 1, 0)
        
        return best_clique_mask

    def partition_graph(adj, n, deadline):
        """
        Iteratively find max clique, remove its edges, repeat.
        """
        # Work on a copy of adj to remove edges
        work_adj = list(adj)
        
        cliques = []
        
        # Optimization: If graph is sparse, max clique might be small.
        # We stop when no edges are left.
        
        while time.time() < deadline:
            # Check if any edges remain
            has_edge = False
            for u in range(n):
                if work_adj[u]:
                    has_edge = True
                    break
            
            if not has_edge:
                break
            
            # Find max clique
            # For very dense graphs, max clique is large.
            # For sparse, it might be 2.
            # If max clique size is 2, we just add edges one by one?
            # No, BK will find a triangle or edge.
            
            clique_mask = max_clique(work_adj, n, deadline)
            
            if clique_mask == 0:
                # Fallback: find any edge
                for u in range(n):
                    if work_adj[u]:
                        lsb = work_adj[u] & -work_adj[u]
                        v = lsb.bit_length() - 1
                        clique_mask = (1 << u) | (1 << v)
                        break
            
            # Extract vertices from clique_mask
            verts = []
            tmp = clique_mask
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                verts.append(v)
                tmp ^= lsb
            
            if len(verts) < 2:
                # Should not happen if has_edge is True
                break
            
            cliques.append(verts)
            
            # Remove all edges within this clique from work_adj
            # For each pair in clique, remove edge
            for i in range(len(verts)):
                u = verts[i]
                for j in range(i + 1, len(verts)):
                    v = verts[j]
                    work_adj[u] &= ~(1 << v)
                    work_adj[v] &= ~(1 << u)
            
            # Safety check: if no progress (e.g. clique size 1 despite edges), break
            # But BK should find size >= 2 if edges exist.
            
        # Fallback: cover any remaining edges as 2-cliques
        # This ensures validity if the loop exits due to time or other issues
        covered = set()
        # We don't track covered in the main loop easily because we modify work_adj.
        # But work_adj is now empty if loop finished normally.
        # If loop broke due to time, work_adj might still have edges.
        
        # Let's do a final pass on work_adj to ensure all edges are covered.
        # We need to be careful not to duplicate edges already in cliques.
        # Since we removed edges from work_adj when adding to cliques,
        # any edge remaining in work_adj was NOT in any clique.
        
        for u in range(n):
            mask = work_adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    cliques.append([u, v])
                    # No need to remove from work_adj if we're just iterating and returning
                mask ^= lsb
        
        return cliques

    start_time = time.time()
    # Leave 10s margin for safety
    deadline = start_time + 140
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    # Process instance 1
    # Allocate time proportionally? n=32 is small, n=64 is bigger.
    # Let's give 20s to first, rest to second?
    # Or just run until deadline for first, then check time for second.
    
    cliques_1 = partition_graph(adj_1, 32, deadline)
    
    # Check time
    if time.time() > deadline - 5:
        # Not enough time for instance 2 properly?
        # Just do a quick greedy for instance 2
        # Or just return what we have? No, we need both.
        # Let's hope instance 1 was fast.
        pass
        
    cliques_2 = partition_graph(adj_2, 64, deadline)
    
    return [cliques_1, cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
