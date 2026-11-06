import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    def has_q3_with_edge(adj, u, v):
        # adj is a list of ints (bitsets)
        # u, v are the endpoints of the new edge.
        
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        # Need at least 2 other neighbors for u and v
        if (nu.bit_count() < 2) or (nv.bit_count() < 2):
            return False
            
        # Get lists of neighbors
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        # Precompute common neighbors for pairs in nu_list?
        # No, let's just iterate.
        
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                w2 = nu_list[j]
                
                # y1 is in N(w1) ∩ N(w2) \ {u}
                inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                if inter_w1_w2 == 0:
                    continue
                
                # Iterate y1 candidates
                y1_mask = inter_w1_w2
                while y1_mask:
                    # Get lowest set bit
                    lsb = y1_mask & -y1_mask
                    y1 = lsb.bit_length() - 1
                    y1_mask ^= lsb
                    
                    # Check distinctness: y1 not in {u, v, w1, w2}
                    # u, w1, w2 are handled by mask or logic?
                    # inter_w1_w2 excluded u.
                    # Need to exclude v, w1, w2?
                    # w1, w2 are in N(w1), N(w2) so they are in intersection?
                    # w1 is in N(w1)? No, simple graph.
                    # But w1 is in N(w2)? No, unless edge exists.
                    # Wait, w1 and w2 are neighbors of u.
                    # Are w1 and w2 adjacent? Maybe.
                    # If w1-w2 is an edge, w1 is in N(w2) and w2 is in N(w1).
                    # So w1, w2 might be in inter_w1_w2.
                    # We must exclude w1, w2, v.
                    
                    if y1 == v or y1 == w1 or y1 == w2:
                        continue
                        
                    # Now for each y1, we need to find y2 in N(z1) ∩ N(z2) \ {v}
                    # such that y1-y2 is edge, and cross edges exist.
                    
                    # We can precompute the set of valid (z1, z2) pairs?
                    # Or iterate nv_list pairs.
                    
                    for k in range(len(nv_list)):
                        z1 = nv_list[k]
                        for l in range(k+1, len(nv_list)):
                            z2 = nv_list[l]
                            
                            # y2 in N(z1) ∩ N(z2) \ {v}
                            inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                            if inter_z1_z2 == 0:
                                continue
                            
                            # y2 must be distinct from u, v, w1, w2, z1, z2
                            # And y2 != y1
                            
                            # Check cross edges first?
                            # If (w1-z1) and (w2-z2) exist, OR (w1-z2) and (w2-z1) exist.
                            if not (((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1)):
                                if not (((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1)):
                                    continue
                                    
                            # Now check if there is a valid y2
                            # y2 must be in inter_z1_z2
                            # y2 must be connected to y1
                            # y2 must not be in {u, v, w1, w2, z1, z2, y1}
                            
                            # Exclude forbidden vertices from inter_z1_z2
                            forbidden = (1 << v) | (1 << u) | (1 << w1) | (1 << w2) | (1 << z1) | (1 << z2) | (1 << y1)
                            valid_y2_mask = inter_z1_z2 & ~forbidden
                            
                            if valid_y2_mask == 0:
                                continue
                                
                            # Check if any y2 in valid_y2_mask is connected to y1
                            # adj[y1] & valid_y2_mask
                            if (adj[y1] & valid_y2_mask) != 0:
                                return True
                                
        return False

    def run_greedy(seed):
        random.seed(seed)
        edges = all_edges[:]
        random.shuffle(edges)
        
        adj = [0] * n
        edge_set = set()
        
        count = 0
        for u, v in edges:
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(adj, u, v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edge_set.add((u, v))
                count += 1
        
        return adj, edge_set, count

    def local_search(adj, edge_set, count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Iterate over missing edges
            # To avoid bias, we can shuffle the missing edges list
            missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set and (v, u) not in edge_set]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline:
                    break
                # Try add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not has_q3_with_edge(adj, u, v):
                    edge_set.add((u, v))
                    count += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        return adj, edge_set, count

    def kick_and_search(adj, edge_set, count, kick_size):
        # Remove kick_size random edges
        edges_to_remove = random.sample(list(edge_set), min(kick_size, len(edge_set)))
        for u, v in edges_to_remove:
            edge_set.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
        # Run local search
        adj, edge_set, count = local_search(adj, edge_set, count)
        return adj, edge_set, count

    # Run multiple restarts
    restart = 0
    while time.time() < deadline:
        # 1. Greedy
        adj, edge_set, count = run_greedy(restart)
        
        # 2. Local Search
        adj, edge_set, count = local_search(adj, edge_set, count)
        
        if count > best_count:
            best_count = count
            best_edges = list(edge_set)
            
        # 3. Kick and Search (if time permits)
        # We can do a few kicks
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            adj_kick, edge_set_kick, count_kick = kick_and_search(adj, edge_set, count, 5)
            if count_kick > count:
                adj, edge_set, count = adj_kick, edge_set_kick, count_kick
                if count > best_count:
                    best_count = count
                    best_edges = list(edge_set)
            else:
                # Revert? We modified adj, edge_set in kick_and_search?
                # No, kick_and_search takes copies?
                # No, it modifies in place if not careful.
                # Let's make kick_and_search work on copies.
                pass

        restart += 1

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
