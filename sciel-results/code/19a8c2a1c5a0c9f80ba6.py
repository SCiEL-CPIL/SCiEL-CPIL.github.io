import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g):
        if n == 0:
            return False
        adj = [[] for _ in range(n)]
        seen_edges = set()
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        max_depth = g // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        if v == parent[curr]:
                            continue
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_luw_graph(q_val, d_val):
        # ... (keep existing LUW code) ...
        pass # Placeholder, will be replaced by full code in edit

    def build_voltage_graph(base_adj, group_order, group_mods, voltages):
        # base_adj: list of lists of (neighbor, edge_index)
        # group_mods: list of moduli for Z_m1 x Z_m2 ...
        # voltages: list of tuples, one per edge in base graph
        
        n_base = len(base_adj)
        n = n_base * group_order
        
        # Map group element to index
        # For Z_5 x Z_4, order 40.
        # Index = a * 4 + b? No, a in 0..4, b in 0..3.
        # Let's define a mapping function.
        
        # Precompute group operations?
        # For small groups, just compute on the fly or precompute table.
        # Here we assume abelian group Z_m1 x ...
        
        def add_group(g1, g2):
            return tuple((g1[i] + g2[i]) % group_mods[i] for i in range(len(group_mods)))
        
        def sub_group(g1, g2):
            return tuple((g1[i] - g2[i]) % group_mods[i] for i in range(len(group_mods)))
            
        def get_index(g):
            idx = 0
            mult = 1
            for i in range(len(group_mods)):
                idx += g[i] * mult
                mult *= group_mods[i]
            return idx
            
        def get_element(idx):
            g = []
            for i in range(len(group_mods)):
                g.append(idx % group_mods[i])
                idx //= group_mods[i]
            return tuple(g)

        edges = []
        # For each base edge, add lifted edges
        # Base edge u-v with voltage v_g
        # Lifted edges: (u, g) -- (v, g + v_g)
        # Note: We must ensure we don't add duplicate edges.
        # Since base graph is simple, and we add for each base edge,
        # and each lifted edge corresponds to a unique base edge and group element,
        # it should be simple if voltages are consistent?
        # Wait, if there are multiple paths, we might get duplicate edges?
        # No, the lift is a covering map. Each edge in base maps to |G| edges in lift.
        # Total edges = |E_base| * |G|.
        
        for u in range(n_base):
            for v, e_idx in base_adj[u]:
                if u > v: # Avoid double processing? No, base_adj is directed for construction?
                    # Let's assume base_adj contains all edges once.
                    # Or we iterate u < v?
                    # My base_adj construction below will be symmetric.
                    # Let's just iterate all u and all neighbors v.
                    # If we add (u,g)-(v, g+v_e) for all u,v, we will add each edge twice?
                    # No, (u,g)-(v, g') and (v,g')-(u,g) are the same edge.
                    # We should only add if u < v in base?
                    # Or add all and deduplicate?
                    # Let's add all and use a set of edge keys.
                    pass
        
        # Better: Iterate base edges once.
        # We need a list of base edges.
        base_edges = []
        seen_base = set()
        for u in range(n_base):
            for v, e_idx in base_adj[u]:
                if (u, v) not in seen_base and (v, u) not in seen_base:
                    if u < v:
                        base_edges.append((u, v, e_idx))
                        seen_base.add((u, v))
                    elif v < u:
                        base_edges.append((v, u, e_idx))
                        seen_base.add((v, u))
        
        # Actually, simpler:
        # Just generate all lifted edges and put in a set.
        lifted_edges_set = set()
        
        for u in range(n_base):
            for v, e_idx in base_adj[u]:
                # For each group element g
                for g_idx in range(group_order):
                    g = get_element(g_idx)
                    # Source vertex: u * group_order + g_idx
                    # Target vertex: v * group_order + index(g + voltage)
                    # Voltage is voltages[e_idx]
                    v_g = voltages[e_idx]
                    new_g = add_group(g, v_g)
                    new_g_idx = get_index(new_g)
                    
                    u_lift = u * group_order + g_idx
                    v_lift = v * group_order + new_g_idx
                    
                    key = (min(u_lift, v_lift), max(u_lift, v_lift))
                    lifted_edges_set.add(key)
                    
        edges = list(lifted_edges_set)
        return n, edges

    # ... rest of code ...
