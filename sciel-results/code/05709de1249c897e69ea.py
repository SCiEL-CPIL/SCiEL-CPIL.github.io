"""XOR-circuit synthesis: Steiner-tree (Prim) seeding + greedy subexpression sharing
with exact GF(2) verification and gate elimination."""
import time
import numpy as np
from collections import deque

_DEADLINE = 45.0  # safety margin under the 60s per-eval / 90s wall-clock limit


def _rows_to_masks(matrix):
    M = np.asarray(matrix, dtype=np.uint8)
    m, n = M.shape
    rows = []
    for i in range(m):
        mask = 0
        r = M[i]
        for j in range(n):
            if r[j]:
                mask |= (1 << j)
        rows.append(mask)
    return rows, m, n


def _min_need(A, targets, n):
    """For each target, min number of elements of A that XOR to it (0 for target 0)."""
    size = 1 << n
    A = np.asarray(A, dtype=np.uint16)
    reached = np.zeros(size, dtype=bool)
    result = {}
    remaining = set()
    for t in targets:
        if t == 0:
            result[0] = 0
        elif t not in result:
            remaining.add(t)
    if not remaining:
        return result
    reached[0] = True
    level = np.array([0], dtype=np.uint16)
    d = 0
    while level.size and remaining:
        d += 1
        nxt = (level[:, None] ^ A[None, :]).ravel()
        nxt = np.unique(nxt)
        new = nxt[~reached[nxt]]
        if new.size:
            reached[nxt] = True
            for mm in new:
                if mm in remaining:
                    result[mm] = d
                    remaining.discard(mm)
            level = new
        else:
            break
    for t in remaining:
        result[t] = 10 ** 9
    return result


def _decompose(A, t, n):
    """BFS: list of elements of A that XOR to t (minimal count)."""
    A = list(A)
    parent = {0: None}
    q = deque([0])
    while q:
        m = q.popleft()
        if m == t:
            break
        for a in A:
            nm = m ^ a
            if nm not in parent:
                parent[nm] = (m, a)
                q.append(nm)
    if t not in parent:
        return None
    elems = []
    cur = t
    while parent[cur] is not None:
        prev, a = parent[cur]
        elems.append(a)
        cur = prev
    return elems


def _verify(matrix, result):
    rows, m, n = _rows_to_masks(matrix)
    val = {}
    for j in range(n):
        val[j] = 1 << j
    val[n] = 0
    depth = {j: 0 for j in range(n + 1)}
    wid = n + 1
    for (a, b) in result['gates']:
        if a not in val or b not in val:
            return False
        val[wid] = val[a] ^ val[b]
        depth[wid] = max(depth[a], depth[b]) + 1
        if depth[wid] > 16:
            return False
        wid += 1
    outs = result['outputs']
    if len(outs) != m:
        return False
    for i in range(m):
        if outs[i] not in val or val[outs[i]] != rows[i]:
            return False
    return True


def _baseline(matrix):
    rows, m, n = _rows_to_masks(matrix)
    gates = []
    outputs = []
    next_id = n + 1
    for t in rows:
        bits = [j for j in range(n) if (t >> j) & 1]
        if not bits:
            outputs.append(n)
            continue
        cur = bits[0]
        for j in bits[1:]:
            gates.append([cur, j])
            cur = next_id
            next_id += 1
        outputs.append(cur)
    return {'gates': gates, 'outputs': outputs}


def _eliminate_gates(gates, outputs, n):
    """
    Eliminate redundant gates by checking if a gate's output is already available.
    Returns new_gates, new_outputs.
    """
    if not gates:
        return gates, outputs

    # Compute wire values
    vals = [1 << i for i in range(n)]
    vals.append(0)
    for g in gates:
        a, b = g
        vals.append(vals[a] ^ vals[b])

    new_gates = []
    wire_remap = list(range(n + 1))
    new_vals = vals[:n + 1]

    for gi in range(len(gates)):
        a, b = gates[gi]
        old_wire = n + 1 + gi
        val = vals[old_wire]

        found_existing = -1
        for wi in range(len(new_vals)):
            if new_vals[wi] == val:
                found_existing = wi
                break

        if found_existing != -1:
            wire_remap[old_wire] = found_existing
        else:
            new_a = wire_remap[a]
            new_b = wire_remap[b]
            new_wire = n + 1 + len(new_gates)
            new_gates.append([new_a, new_b])
            new_vals.append(val)
            wire_remap[old_wire] = new_wire

    new_outputs = []
    for o in outputs:
        if o < len(wire_remap):
            new_outputs.append(wire_remap[o])
        else:
            new_outputs.append(n)

    return new_gates, new_outputs


def _steiner_mst_nodes(rows, n, deadline_check):
    """
    Steiner-tree heuristic via Prim's algorithm on the reachable subcube.
    Terminals = {0} ∪ rows.  Edge weight = Hamming distance.
    Returns a list of intermediate node masks (in a useful order) that the
    MST implicitly requires as XOR intermediates.
    """
    terminals = sorted(set([0] + rows))
    if len(terminals) <= 1:
        return []

    # BFS over the GF(2) subcube spanned by the n input basis vectors.
    # For n <= 16 the full subcube is at most 65536 elements, which is fine.
    # We compute the set of all reachable vectors (which is the full space 0..2^n-1
    # since the basis vectors span everything).  For n <= 16 we can just use
    # the full space; for larger n we cap the BFS.
    full_size = 1 << n
    if full_size <= 65536:
        # All vectors in the space are reachable.
        pass  # we'll work with the full space implicitly
    else:
        # For very large n (unlikely given the task's 8x8/12x16/16x16),
        # fall back to just the terminals + their pairwise XORs.
        reachable = set(terminals)
        for i in range(len(terminals)):
            for j in range(i + 1, len(terminals)):
                reachable.add(terminals[i] ^ terminals[j])
        # Use reachable as our working set
        nodes = sorted(reachable)
        node_idx = {v: i for i, v in enumerate(nodes)}
        t_idx = [node_idx[t] for t in terminals]
    if full_size <= 65536:
        nodes = list(range(full_size))
        node_idx = {v: i for i, v in enumerate(nodes)}
        t_idx = [node_idx[t] for t in terminals]

    N = len(nodes)
    T = len(t_idx)

    if T <= 1:
        return []

    # Compute Hamming distances between terminals.
    # For the full space we can compute popcount(v) = number of 1-bits.
    # Hamming(u, v) = popcount(u ^ v).
    # Build the terminal-distance matrix.
    term_nodes = [nodes[i] for i in t_idx]
    dist = np.zeros((T, T), dtype=np.int32)
    for i in range(T):
        for j in range(i + 1, T):
            d = bin(term_nodes[i] ^ term_nodes[j]).count('1')
            dist[i, j] = d
            dist[j, i] = d

    # Prim's MST over terminals.
    in_mst = [False] * T
    min_dist = [10**9] * T
    parent = [-1] * T
    in_mst[0] = True
    min_dist[0] = 0
    for _ in range(T):
        # Find closest non-MST terminal
        best = -1
        bd = 10**9
        for i in range(T):
            if not in_mst[i] and min_dist[i] < bd:
                bd = min_dist[i]
                best = i
        if best == -1:
            break
        in_mst[best] = True
        for j in range(T):
            if not in_mst[j] and dist[best, j] < min_dist[j]:
                min_dist[j] = dist[best, j]
                parent[j] = best

    # Collect MST edges (parent links).
    mst_edges = []
    for i in range(T):
        if parent[i] != -1:
            mst_edges.append((parent[i], i))

    # For each MST edge (u, v), the "Steiner node" that connects them in the
    # hypercube is not unique, but a natural candidate is any vector w such that
    # w is "between" u and v in Hamming space.  The simplest useful intermediate
    # is: for each pair of terminals connected in the MST, consider the XOR of
    # their difference with various existing nodes.  A practical heuristic:
    # the intermediate node for edge (u,v) is the vector that is the XOR of
    # u and v "reduced" — but in GF(2) the XOR u^v is the difference vector.
    # Actually, in the hypercube, the path from u to v goes through intermediate
    # vertices.  A key intermediate is: take u, and flip one bit that differs
    # from v.  But we want nodes that are useful for *multiple* targets.
    #
    # Better approach: the MST tells us which pairs of terminals are "connected".
    # For each MST edge, the XOR of the two terminals is a candidate intermediate
    # (it's the vector you'd need to XOR one terminal with to get the other).
    # More generally, any vector w such that w = a ^ b for some available a, b
    # is a candidate.  The MST edges suggest which XORs are most valuable.
    #
    # Heuristic: for each MST edge (u, v), add the vector (term_nodes[u] ^ term_nodes[v])
    # as a candidate intermediate.  Also, for each terminal t, for each MST-neighbor,
    # the XOR is a useful intermediate.
    candidates = set()
    for (ui, vi) in mst_edges:
        u_node = term_nodes[ui]
        v_node = term_nodes[vi]
        xor_val = u_node ^ v_node
        if xor_val != 0:
            candidates.add(xor_val)

    # Additionally, for each terminal, the XOR with each other terminal in the MST
    # (transitively) gives useful intermediates.  But the edge XORs above already
    # capture the direct connections.  Let's also add: for each candidate c,
    # for each terminal t, t ^ c is another candidate (one more XOR step).
    # This is a 2-level expansion.
    expanded = set(candidates)
    for c in list(candidates):
        for t in term_nodes:
            v = t ^ c
            if v != 0 and v not in expanded:
                expanded.add(v)

    # Filter: only keep candidates that are not already terminals and are
    # actually "useful" (reduce the need for at least one target).
    terminal_set = set(term_nodes)
    useful = []
    for c in sorted(expanded):
        if c in terminal_set:
            continue
        # Check if c helps: for some target t, popcount(t ^ c) < popcount(t)
        helpful = False
        for t in term_nodes:
            if bin(t ^ c).count('1') < bin(t).count('1'):
                helpful = True
                break
        if helpful:
            useful.append(c)

    # Sort by "usefulness": how many targets it helps, descending.
    def usefulness(c):
        cnt = 0
        for t in term_nodes:
            if bin(t ^ c).count('1') < bin(t).count('1'):
                cnt += 1
        return -cnt

    useful.sort(key=usefulness)

    # Cap the number of Steiner nodes to avoid blowing up the search.
    max_steiner = 40
    return useful[:max_steiner]


def synthesize(matrix, depth_limit):
    t0 = time.time()
    rows, m, n = _rows_to_masks(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}

    wire_mask = {}
    mask_id = {}
    depth = {}
    for j in range(n):
        wire_mask[j] = 1 << j
        mask_id[1 << j] = j
        depth[j] = 0
    wire_mask[n] = 0
    mask_id[0] = n
    depth[n] = 0

    gates = []
    next_id = [n + 1]

    def add_gate(a, b):
        c = wire_mask[a] ^ wire_mask[b]
        if c in mask_id:
            return mask_id[c]
        wid = next_id[0]
        next_id[0] += 1
        wire_mask[wid] = c
        depth[wid] = max(depth[a], depth[b]) + 1
        mask_id[c] = wid
        gates.append((a, b))
        return wid

    A_ids = list(range(n + 1))
    A_masks = [wire_mask[i] for i in A_ids]
    targets = rows

    def cur_future(need):
        return sum(max(0, need[t] - 1) for t in targets)

    K = 50 if n >= 14 else 80
    max_gates = 250

    # ---- Steiner-tree (Prim) seeding ----
    if time.time() - t0 < _DEADLINE:
        steiner_nodes = _steiner_mst_nodes(rows, n, lambda: time.time() - t0 < _DEADLINE)
        for c in steiner_nodes:
            if time.time() - t0 > _DEADLINE:
                break
            if c in mask_id:
                continue
            # Find a valid producing pair for c
            found = False
            L = len(A_ids)
            for i in range(L):
                complement = A_masks[i] ^ c
                if complement in mask_id:
                    jid = mask_id[complement]
                    if jid in depth:
                        a, b = A_ids[i], jid
                        d = max(depth[a], depth[b]) + 1
                        if d <= depth_limit:
                            add_gate(a, b)
                            A_ids.append(next_id[0] - 1)
                            A_masks.append(c)
                            found = True
                            break
                if found:
                    break
            if not found:
                continue

    # ---- greedy subexpression sharing ----
    while time.time() - t0 < _DEADLINE and (next_id[0] - (n + 1)) < max_gates:
        need = _min_need(A_masks, targets, n)
        if all(need[t] <= 1 for t in targets):
            break
        cur_fut = cur_future(need)

        # build candidate pool
        pool = {}
        L = len(A_ids)
        for i in range(L):
            mi = A_masks[i]
            di = depth[A_ids[i]]
            for j in range(i + 1, L):
                c = mi ^ A_masks[j]
                if c in mask_id:
                    continue
                d = max(di, depth[A_ids[j]]) + 1
                if d <= depth_limit:
                    pool[c] = (A_ids[i], A_ids[j])
        # target-adjacent candidates
        for t in targets:
            for i in range(L):
                c = t ^ A_masks[i]
                if c in mask_id:
                    continue
                d = depth[A_ids[i]] + 1
                if d <= depth_limit:
                    pool[c] = (A_ids[i], c)

        cand = []
        for c, (a, b) in pool.items():
            if wire_mask.get(a, -1) ^ wire_mask.get(b, -1) != c:
                found = False
                for i in range(L):
                    if A_masks[i] ^ c in mask_id:
                        jid = mask_id[A_masks[i] ^ c]
                        if jid in depth:
                            a, b = A_ids[i], jid
                            found = True
                            break
                if not found:
                    continue
            d = max(depth[a], depth[b]) + 1
            if d > depth_limit:
                continue
            sc = 0
            for t in targets:
                wt = bin(t).count('1')
                wt2 = bin(t ^ c).count('1')
                if wt2 < wt:
                    sc += (wt - wt2)
            cand.append((sc, c, a, b))
        if not cand:
            break
        cand.sort(key=lambda x: -x[0])
        cand = cand[:K]

        best = None
        best_cost = None
        for (sc, c, a, b) in cand:
            if time.time() - t0 > _DEADLINE:
                break
            need2 = _min_need(A_masks + [c], targets, n)
            cost = 1 + cur_future(need2)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = (c, a, b)
        if best is None or best_cost >= cur_fut:
            break
        c, a, b = best
        add_gate(a, b)
        A_ids.append(next_id[0] - 1)
        A_masks.append(c)

    # ---- produce any target not yet an available wire (depth-safe) ----
    need = _min_need(A_masks, targets, n)
    for t in targets:
        if need.get(t, 10 ** 9) <= 1:
            continue
        done = False
        elems = _decompose(A_masks, t, n)
        if elems is not None:
            maxd = max(depth[mask_id[e]] for e in elems)
            if (len(elems) - 1) + maxd <= 16:
                cur = None
                for e in elems:
                    eid = mask_id[e]
                    if cur is None:
                        cur = eid
                    else:
                        cur = add_gate(cur, eid)
                mask_id[t] = cur
                done = True
        if not done:
            bits = [j for j in range(n) if (t >> j) & 1]
            if not bits:
                mask_id[t] = n
            else:
                cur = bits[0]
                for j in bits[1:]:
                    cur = add_gate(cur, j)
                mask_id[t] = cur

    outputs = [mask_id[t] for t in targets]

    result = {'gates': [[a, b] for (a, b) in gates], 'outputs': outputs}

    # Gate elimination
    new_gates, new_outputs = _eliminate_gates(result['gates'], result['outputs'], n)
    result = {'gates': new_gates, 'outputs': new_outputs}

    if not _verify(matrix, result):
        result = _baseline(matrix)
    return result
