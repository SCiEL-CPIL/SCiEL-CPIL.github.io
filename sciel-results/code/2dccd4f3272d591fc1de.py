import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]

    best_edges = []
    best_count = 0

    # Fast incremental Q3 check:
    # Adding edge (u,v) to a Q3-free graph creates a Q3 iff there exists a Q3 containing (u,v).
    # In Q3, u and v are adjacent. N_Q3(u) = {v, w1, w2}, N_Q3(v) = {u, z1, z2}.
    # w1, w2 are distance 2 from u (via y1), z1, z2 are distance 2 from v (via y2).
    # Edges: u-w1, u-w2, v-z1, v-z2, w1-y1, w2-y1, z1-y2, z2-y2,
    #        y1-y2, and either (w1-z1, w2-z2) or (w1-z2, w2-z1).
    def has_q3_with_edge(adj, u, v):
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

        if nu.bit_count() < 2 or nv.bit_count() < 2:
            return False

        # Build neighbor lists
        nu_list = []
        tmp = nu
        while tmp:
            lsb = tmp & (-tmp)
            nu_list.append(lsb.bit_length() - 1)
            tmp ^= lsb

        nv_list = []
        tmp = nv
        while tmp:
            lsb = tmp & (-tmp)
            nv_list.append(lsb.bit_length() - 1)
            tmp ^= lsb

        # Iterate over pairs (w1, w2) from N(u) \ {v}
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i + 1, len(nu_list)):
                w2 = nu_list[j]

                # y1 must be in N(w1) ∩ N(w2) \ {u}
                inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                if inter_w1_w2 == 0:
                    continue

                # Iterate over y1 candidates
                tmp = inter_w1_w2
                while tmp:
                    lsb = tmp & (-tmp)
                    y1 = lsb.bit_length() - 1
                    tmp ^= lsb

                    # y1 must not be u, v, w1, w2
                    if y1 in (u, v, w1, w2):
                        continue

                    # Iterate over pairs (z1, z2) from N(v) \ {u}
                    for k in range(len(nv_list)):
                        z1 = nv_list[k]
                        if z1 == y1:
                            continue
                        for l in range(k + 1, len(nv_list)):
                            z2 = nv_list[l]
                            if z2 == y1 or z2 == w1 or z2 == w2:
                                continue
                            if z1 == w1 or z1 == w2:
                                continue

                            # y2 must be in N(z1) ∩ N(z2) \ {v}
                            inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                            if inter_z1_z2 == 0:
                                continue

                            tmp2 = inter_z1_z2
                            while tmp2:
                                lsb2 = tmp2 & (-tmp2)
                                y2 = lsb2.bit_length() - 1
                                tmp2 ^= lsb2

                                # Distinctness checks
                                if y2 in (u, v, w1, w2, z1, z2, y1):
                                    continue

                                # Check y1-y2 edge
                                if not (adj[y1] >> y2) & 1:
                                    continue

                                # Check cross edges: (w1-z1 & w2-z2) or (w1-z2 & w2-z1)
                                if ((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1):
                                    return True
                                if ((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1):
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
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

            if has_q3_with_edge(adj, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edge_set.add((u, v))
                count += 1

        # Local search: try adding missing edges
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u, v in all_edges:
                if (u, v) in edge_set:
                    continue
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

        return count, list(edge_set)

    # Multiple restarts with different seeds
    restart = 0
    while time.time() < deadline - 5:
        count, edges_list = run_greedy(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list

        restart += 1

    return [[u, v] for u, v in best_edges]
