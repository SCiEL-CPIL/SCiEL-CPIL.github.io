import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    A = np.array(matrix, dtype=np.int8)

    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}

    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        for i in range(m):
            expected = 0
            for j in range(n):
                if A[i][j]:
                    expected |= (1 << j)
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        return max((wire_depth[w] for w in outputs if w < num_wires), default=0)

    def _baseline_circuit():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            row = A[i]
            if not np.any(row):
                outputs.append(n)
            else:
                indices = np.where(row == 1)[0].tolist()
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}

    def _steinersyn(A_mat, time_budget):
        """
        Steiner-tree based XOR synthesis.
        """
        m_c = A_mat.shape[0]
        n_c = A_mat.shape[1]

        # Build candidate pool
        pool = []
        for i in range(n_c):
            pool.append(1 << i)
        pool.append(0)
        for i in range(m_c):
            bits = 0
            for j in range(n_c):
                if A_mat[i][j]:
                    bits |= (1 << j)
            pool.append(bits)

        # Add pairwise XORs (limited)
        seen = set(pool)
        pool_size = len(pool)
        max_pool = 150
        for i in range(pool_size):
            if len(pool) >= max_pool:
                break
            for j in range(i + 1, pool_size):
                if len(pool) >= max_pool:
                    break
                x = pool[i] ^ pool[j]
                if x not in seen:
                    seen.add(x)
                    pool.append(x)

        P = len(pool)
        if P < 2:
            return None

        # Terminals: zero + target rows
        term_indices = [n_c] + list(range(n_c + 1, n_c + 1 + m_c))

        # Compute weight matrix (Hamming distance)
        weights = [[0] * P for _ in range(P)]
        for i in range(P):
            pi = pool[i]
            for j in range(i + 1, P):
                d = (pi ^ pool[j]).bit_count()
                weights[i][j] = d
                weights[j][i] = d

        # Prim's MST on full graph
        in_mst = [False] * P
        min_w = [float('inf')] * P
        min_from = [-1] * P
        mst_edges = []

        start = n_c
        in_mst[start] = True
        for _ in range(P - 1):
            if time.time() > time_budget:
                return None
            best_v = -1
            best_w = float('inf')
            for v in range(P):
                if not in_mst[v] and min_w[v] < best_w:
                    best_w = min_w[v]
                    best_v = v
            if best_v == -1:
                break
            in_mst[best_v] = True
            if min_from[best_v] != -1:
                mst_edges.append((min_from[best_v], best_v))
            for v in range(P):
                if not in_mst[v] and weights[best_v][v] < min_w[v]:
                    min_w[v] = weights[best_v][v]
                    min_from[v] = best_v

        # Extract Steiner subtree (union of paths from root to each terminal)
        adj = [[] for _ in range(P)]
        for u, v in mst_edges:
            adj[u].append(v)
            adj[v].append(u)

        root = term_indices[0]
        parent = [-1] * P
        parent[root] = -2
        order = []
        stack = [root]
        while stack:
            node = stack.pop()
            order.append(node)
            for nb in adj[node]:
                if parent[nb] == -1:
                    parent[nb] = node
                    stack.append(nb)

        in_steiner = [False] * P
        for t in term_indices:
            node = t
            while node != -2:
                in_steiner[node] = True
                if node == root:
                    break
                node = parent[node]

        # Build Steiner tree adjacency
        s_adj = [[] for _ in range(P)]
        for u, v in mst_edges:
            if in_steiner[u] and in_steiner[v]:
                s_adj[u].append(v)
                s_adj[v].append(u)

        # Root Steiner tree
        s_parent = [-1] * P
        s_parent[root] = -2
        s_order = []
        stack = [root]
        while stack:
            node = stack.pop()
            s_order.append(node)
            for nb in s_adj[node]:
                if s_parent[nb] == -1:
                    s_parent[nb] = node
                    stack.append(nb)

        # Synthesize circuit
        gates = []
        next_wire = n_c + 1
        wire_for = {}
        wire_vec = {}

        for i in range(n_c):
            wire_for[i] = i
            wire_vec[i] = 1 << i
        wire_for[n_c] = n_c
        wire_vec[n_c] = 0

        for node in s_order:
            if node in wire_for:
                continue
            vec = pool[node]
            if vec == 0:
                wire_for[node] = n_c
                continue
            # Compute vec from inputs
            set_bits = []
            temp = vec
            bit = 0
            while temp:
                if temp & 1:
                    set_bits.append(bit)
                temp >>= 1
                bit += 1
            if len(set_bits) == 1:
                wire_for[node] = set_bits[0]
            else:
                prev = set_bits[0]
                for b in set_bits[1:]:
                    gates.append([prev, b])
                    prev = next_wire
                    next_wire += 1
                wire_for[node] = prev

        # Map target rows to wires
        outputs = []
        for i in range(m_c):
            outputs.append(wire_for[n_c + 1 + i])

        circuit = {'gates': gates, 'outputs': outputs}
        return circuit

    # Main
    best_result = _baseline_circuit()
    best_cost = len(best_result['gates'])

    # Try Steiner approach
    circuit = _steinersyn(A, deadline - 5.0)
    if circuit is not None:
        cost = len(circuit['gates'])
        if cost < best_cost:
            if _verify(circuit):
                d = _compute_depth(circuit['gates'], circuit['outputs'])
                if d <= depth_limit:
                    best_cost = cost
                    best_result = circuit

    return best_result
