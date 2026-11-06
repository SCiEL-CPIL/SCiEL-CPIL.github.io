import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    d = 7
    n_vertices = 2 ** d
    n_edges = n_vertices * d // 2  # 448
    
    # Edge representation: (u, v) with u < v
    # Map edge to index
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx += 1
    
    # Generate all C4s
    # A C4 is defined by 2 bits i, j and a vertex u that doesn't have those bits set?
    # No, a C4 in Q_d is a set of 4 vertices {u, u^i, u^j, u^i^j} where i,j are distinct bits.
    # The edges are (u, u^i), (u, u^j), (u^i, u^i^j), (u^j, u^i^j).
    # We need to ensure we don't double count.
    # We can iterate over all pairs of bits (i, j) and all "base" vertices u such that 
    # the 4 vertices are distinct. They are always distinct if i!=j.
    # To avoid duplicates, we can fix a canonical form for the C4.
    # Or just generate all and use a set of frozensets of edge indices.
    
    c4s = []
    c4_set = set()
    for i in range(d):
        for j in range(i+1, d):
            bit_i = 1 << i
            bit_j = 1 << j
            # Iterate over all vertices u. 
            # The C4 involves u, u^bit_i, u^bit_j, u^bit_i^bit_j.
            # To get unique C4s, we can require that u does not have bit_i or bit_j set?
            # No, that restricts which vertices are in the C4.
            # Actually, the set of 4 vertices is unique regardless of which one we call "u".
            # So we can just iterate u from 0 to 2^d - 1.
            # But we will generate the same C4 4 times (once for each vertex as base).
            # And for each base, the edge set is the same.
            # So we can just add to a set.
            
            for u in range(n_vertices):
                v1 = u
                v2 = u ^ bit_i
                v3 = u ^ bit_j
                v4 = u ^ bit_i ^ bit_j
                
                # The 4 edges
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                # Get indices
                try:
                    idx1 = edge_to_idx[e1]
                    idx2 = edge_to_idx[e2]
                    idx3 = edge_to_idx[e3]
                    idx4 = edge_to_idx[e4]
                except KeyError:
                    continue # Should not happen
                
                c4_tuple = (idx1, idx2, idx3, idx4)
                # Sort to make hashable unique
                c4_sorted = tuple(sorted(c4_tuple))
                if c4_sorted not in c4_set:
                    c4_set.add(c4_sorted)
                    c4s.append(c4_sorted)
    
    n_c4s = len(c4s)
    
    # Precompute for each edge, which C4s it hits
    edge_to_c4s = [[] for _ in range(n_edges)]
    for c4_idx, c4 in enumerate(c4s):
        for e_idx in c4:
            edge_to_c4s[e_idx].append(c4_idx)
    
    # Greedy initialization of R (removal set)
    # We want to hit all C4s.
    cover_count = [0] * n_c4s
    R = set()
    
    # Greedy: pick edge that covers most uncovered C4s
    uncovered_c4s = set(range(n_c4s))
    while uncovered_c4s:
        best_edge = -1
        best_count = -1
        # This loop is O(E * C4s_per_edge) per iteration, which is slow if done naively.
        # But E=448, C4s=672.
        # Let's just iterate edges and count how many uncovered C4s they hit.
        for e in range(n_edges):
            if e in R:
                continue
            count = 0
            for c4_idx in edge_to_c4s[e]:
                if c4_idx in uncovered_c4s:
                    count += 1
            if count > best_count:
                best_count = count
                best_edge = e
        
        if best_edge == -1 or best_count == 0:
            break # No edge can cover remaining? Should not happen.
        
        R.add(best_edge)
        for c4_idx in edge_to_c4s[best_edge]:
            if c4_idx in uncovered_c4s:
                uncovered_c4s.remove(c4_idx)
                cover_count[c4_idx] += 1
            else:
                cover_count[c4_idx] += 1 # Update count even if already covered? 
                # Wait, cover_count tracks total hits from R.
                # We need cover_count to reflect current R.
        
    # Re-calculate cover_count to be sure
    cover_count = [0] * n_c4s
    for e in R:
        for c4_idx in edge_to_c4s[e]:
            cover_count[c4_idx] += 1
            
    current_size_R = len(R)
    best_size_R = current_size_R
    best_R = set(R)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # Helper to check if R is valid (all cover_count > 0)
    # We maintain cover_count, so we just need to check if any is 0.
    # But we update incrementally.
    
    def is_valid():
        # Check if any C4 has cover_count 0
        # This is O(N_C4). Can be optimized by tracking number of uncovered.
        # Let's track uncovered_count.
        pass
    
    # Let's track uncovered_count for speed
    uncovered_count = sum(1 for c in cover_count if c == 0)
    
    while time.time() < deadline and T > min_T:
        if random.random() < 0.5:
            # Try Remove
            if R:
                e_out = random.choice(list(R))
                # Check if removing e_out keeps validity
                # It keeps validity if for all c4 in edge_to_c4s[e_out], cover_count[c4] > 1
                # i.e., cover_count[c4] - 1 > 0
                can_remove = True
                for c4_idx in edge_to_c4s[e_out]:
                    if cover_count[c4_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Accept removal
                    R.remove(e_out)
                    for c4_idx in edge_to_c4s[e_out]:
                        cover_count[c4_idx] -= 1
                        if cover_count[c4_idx] == 0:
                             # Should not happen if can_remove was true
                             pass
                    # Update uncovered_count
                    # We decremented counts. If any went from 1 to 0, uncovered_count increases.
                    # But can_remove ensured no count was 1. So no count became 0.
                    # So uncovered_count remains 0.
                    
                    current_size_R -= 1
                    if current_size_R < best_size_R:
                        best_size_R = current_size_R
                        best_R = set(R)
                else:
                    # Try Swap
                    if len(R) > 0:
                        e_in = random.randint(0, n_edges - 1)
                        if e_in in R:
                            continue
                        # Check if swap keeps validity
                        # Remove e_out, Add e_in.
                        # A C4 becomes uncovered if:
                        # 1. It was hit by e_out but not e_in, and cover_count was 1.
                        # 2. It was not hit by e_out, but was hit by e_in? No, adding helps.
                        # So we only need to check C4s hit by e_out.
                        can_swap = True
                        # Pre-check: does e_in hit any C4 that e_out hits?
                        # We need to check specific condition.
                        e_in_c4s_set = set(edge_to_c4s[e_in])
                        for c4_idx in edge_to_c4s[e_out]:
                            if cover_count[c4_idx] == 1 and c4_idx not in e_in_c4s_set:
                                can_swap = False
                                break
                        
                        if can_swap:
                            # Perform swap
                            R.remove(e_out)
                            R.add(e_in)
                            
                            # Update cover_count
                            for c4_idx in edge_to_c4s[e_out]:
                                cover_count[c4_idx] -= 1
                            for c4_idx in edge_to_c4s[e_in]:
                                cover_count[c4_idx] += 1
                            
                            # Size unchanged
                        else:
                            # Reject swap
                            pass
        else:
            # Try Add (to escape local minima, though it worsens objective)
            # We only accept if it allows future improvements? 
            # In SA, we accept worse states with prob exp(-Delta/T).
            # Delta = +1 (size increases by 1).
            # Probability = exp(-1/T).
            if T > 0.1: # Only do this when T is high enough
                e_in = random.randint(0, n_edges - 1)
                if e_in not in R:
                    # Check if adding e_in causes any C4 to be "over-covered"? No, validity is about hitting.
                    # Adding an edge never breaks validity (it only adds hits).
                    # So adding is always valid.
                    # But it increases |R|.
                    # Accept with prob exp(-1/T)
                    if random.random() < pow(2.718, -1.0/T):
                        R.add(e_in)
                        for c4_idx in edge_to_c4s[e_in]:
                            cover_count[c4_idx] += 1
                        current_size_R += 1
        
        T *= cooling_rate
    
    # Final Polish: Greedy removal
    # Try to remove any edge that can be removed
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for e in list(R):
            can_remove = True
            for c4_idx in edge_to_c4s[e]:
                if cover_count[c4_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                R.remove(e)
                for c4_idx in edge_to_c4s[e]:
                    cover_count[c4_idx] -= 1
                current_size_R -= 1
                changed = True
    
    if current_size_R < best_size_R:
        best_R = set(R)
    
    # Construct final edge list
    # Valid edges are those NOT in best_R
    valid_edges = []
    for i, (u, v) in enumerate(edges):
        if i not in best_R:
            valid_edges.append([u, v])
            
    return valid_edges

# Run construct
# The function construct() is the entry point.

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
