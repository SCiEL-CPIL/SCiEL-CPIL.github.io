"""Assembly line balancing via critical-ratio priority with a deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_SECONDS = 115.0  # safety margin before the 120s kill


def priority(instance):
    """
    Return one score per task. The greedy station-filler picks the highest-priority
    eligible task that fits the remaining cycle time.

    Strategy: Critical Ratio (CR) = (1 + sum of successor times) / (task time + 1)
    This favors tasks that are 'critical' in the sense that they have large downstream
    impact relative to their own duration, which tends to reduce idle time in later
    stations.
    """
    t0 = time.time()

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = float(instance["cycle_time"])
    predecessors = instance["predecessors"]

    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            successors[j].append(i)

    # Compute longest path from each node to any sink (downstream impact)
    # longest[i] = max over successors s of (times[s] + longest[s]), or 0 if no successors
    # We need topological order. Since predecessors[i] gives immediate predecessors,
    # we can do a reverse topological sort.
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in predecessors[i]:
            in_degree[i] += 1

    # Kahn's algorithm for topological order
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # If topological sort didn't cover all nodes (cycle), fall back to times
    if len(topo_order) < n:
        return times.copy()

    # Compute longest downstream path in reverse topological order
    longest = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if successors[node]:
            longest[node] = max(times[s] + longest[s] for s in successors[node])
        else:
            longest[node] = 0.0

    # Critical ratio: (1 + longest[i]) / (times[i] + 1)
    # Adding 1 avoids division by zero and smooths the ratio
    cr = (1.0 + longest) / (times + 1.0)

    # Check deadline
    if time.time() - t0 > _DEADLINE_SECONDS:
        return cr

    return cr
# EVOLVE-BLOCK-END
