"""Communication-minimizing DAG pebbling: greedy topological with smart eviction."""
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """Return list of [operation, vertex] that pebbles the DAG.

    dag: object with attributes (or dict keys):
      - parents: list where parents[i] is list of parent (predecessor) vertex indices of vertex i
      - outputs: list of output vertex indices (must finish blue)
      - (optionally) n / num_vertices
    fast_memory_budget: int, number of red values that may reside simultaneously.

    Operations: load (blue->red, costs 1 transfer), store (red->blue, costs 1),
    compute (parents red -> vertex red, costs 0), delete (red->gone, costs 0).
    Inputs start blue. Every output must finish blue.
    """
    deadline = time.time() + 80.0

    # ---- Normalize dag ----
    if isinstance(dag, dict):
        parents = dag.get('parents', dag.get('preds', dag.get('ancestors', [])))
        outputs = dag.get('outputs', dag.get('output', []))
        n = dag.get('n', dag.get('num_vertices', dag.get('num_nodes', len(parents))))
    else:
        parents = getattr(dag, 'parents', getattr(dag, 'preds', getattr(dag, 'ancestors', [])))
        outputs = getattr(dag, 'outputs', getattr(dag, 'output', []))
        n = getattr(dag, 'n', getattr(dag, 'num_vertices', getattr(dag, 'num_nodes', len(parents))))

    if parents is None:
        parents = []
    if outputs is None:
        outputs = []
    if n is None or n == 0:
        n = len(parents)

    parents = [list(p) for p in parents]
    while len(parents) < n:
        parents.append([])

    budget = int(fast_memory_budget)
    if budget < 0:
        budget = 0

    # ---- Build children / topological order ----
    children = defaultdict(list)
    indegree = [0] * n
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            indegree[v] += 1

    # Topological sort (Kahn)
    topo = []
    q = [v for v in range(n) if indegree[v] == 0]
    idx = 0
    while idx < len(q):
        v = q[idx]
        idx += 1
        topo.append(v)
        for c in children[v]:
            indegree[c] -= 1
            if indegree[c] == 0:
                q.append(c)
    if len(topo) < n:
        for v in range(n):
            if v not in set(topo):
                topo.append(v)

    # ---- Determine layers ----
    layer_of = [-1] * n
    for v in topo:
        if not parents[v]:
            layer_of[v] = 0
        else:
            layer_of[v] = max(layer_of[p] for p in parents[v]) + 1
    max_layer = max(layer_of) if n > 0 else 0

    # Group vertices by layer
    layers = defaultdict(list)
    for v in range(n):
        layers[layer_of[v]].append(v)

    output_set = set(outputs)
    is_output = [v in output_set for v in range(n)]

    ops = []
    if n == 0:
        return ops

    # State
    computed = set()
    red = set()
    remaining_children = [len(children[v]) for v in range(n)]

    def is_blue(v):
        return v in computed and v not in red

    def load(v):
        if v not in computed or v in red:
            return False
        if len(red) >= budget:
            return False
        red.add(v)
        ops.append(['load', v])
        return True

    def store(v):
        if v not in red:
            return False
        red.discard(v)
        ops.append(['store', v])
        return True

    def delete(v):
        if v not in red:
            return False
        red.discard(v)
        ops.append(['delete', v])
        return True

    def compute(v):
        if v in computed:
            return False
        for p in parents[v]:
            if p not in red:
                return False
        if len(red) >= budget:
            return False
        computed.add(v)
        red.add(v)
        ops.append(['compute', v])
        return True

    def free_slot():
        """Free one slot in red by deleting or storing a vertex."""
        if not red:
            return False
        # Priority 1: delete vertices with no remaining children and not output
        for candidate in list(red):
            if not is_output[candidate] and remaining_children[candidate] == 0:
                delete(candidate)
                return True
        # Priority 2: store vertex with fewest remaining children
        candidates = list(red)
        candidates.sort(key=lambda c: (remaining_children[c], layer_of[c]))
        if candidates:
            store(candidates[0])
            return True
        return False

    # Process vertices in topological order
    for v in topo:
        if time.time() > deadline:
            break

        # Ensure all parents of v are red
        for p in parents[v]:
            if p not in red:
                if p in computed:
                    # p is blue, need to load
                    while len(red) >= budget:
                        if not free_slot():
                            break
                    load(p)
                # If p not computed, it's an error in topo order

        # Compute v
        if v not in computed:
            all_parents_red = all(p in red for p in parents[v])
            if all_parents_red:
                while len(red) >= budget:
                    if not free_slot():
                        break
                compute(v)

        # Update remaining_children for parents of v
        for p in parents[v]:
            remaining_children[p] -= 1

    # Ensure all outputs are blue
    for v in outputs:
        if v in red:
            store(v)

    # Clean up remaining red vertices
    for v in list(red):
        if v not in output_set:
            delete(v)

    return ops

# EVOLVE-BLOCK-END
