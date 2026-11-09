import itertools
import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 118

    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    all_edges_set = set(all_edges)
    m = len(all_edges)

    # Precompute all 8-sets and their (A,B) splits
    all_8sets = list(itertools.combinations(range(n), 8))
    splits_data = []
    for combo in all_8sets:
        for A_idx in itertools.combinations(range(8), 4):
            B_idx = tuple(x for x in range(8) if x not in A_idx)
            A = tuple(combo[i] for i in A_idx)
            B = tuple(combo[i] for i in B_idx)
            cross_edges = tuple((min(a, b), max(a, b)) for a in A for b in B)
            splits_data.append((A, B, cross_edges))

    def check_q3_exists(adj):
        """Return True if Q3 exists."""
        for A, B, cross_edges in splits_data:
            present = 0
            for (u, v) in cross_edges:
                if (adj[u] >> v) & 1:
                    present += 1
            if present < 12:
                continue
            if present > 12:
                return True
            missing = []
            for (u, v) in cross_edges:
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
            a_seen = set()
            b_seen = set()
            ok = True
            for (u, v) in missing:
                if u in A:
                    if u in a_seen or v in b_seen:
                        ok = False
                        break
                    a_seen.add(u)
                    b_seen.add(v)
                else:
                    if v in a_seen or u in b_seen:
                        ok = False
                        break
                    a_seen.add(v)
                    b_seen.add(u)
            if ok:
                return True
        return False

    def find_q3_edge_to_remove(adj):
        """Find a Q3 and return an edge in it. Returns None if no Q3."""
        for A, B, cross_edges in splits_data:
            present = 0
            present_edges = []
            for (u, v) in cross_edges:
                if (adj[u] >> v) & 1:
                    present += 1
                    present_edges.append((u, v))
            if present < 12:
                continue
            if present > 12:
                return random.choice(present_edges)
            missing = []
            for (u, v) in cross_edges:
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
            a_seen = set()
            b_seen = set()
            ok = True
            for (u, v) in missing:
                if u in A:
                    if u in a_seen or v in b_seen:
                        ok = False
                        break
                    a_seen.add(u)
                    b_seen.add(v)
                else:
                    if v in a_seen or u in b_seen:
                        ok = False
                        break
                    a_seen.add(v)
                    b_seen.add(u)
            if ok:
                return random.choice(present_edges)
        return None

    def count_q3_occurrences(adj):
        """Count how many times each edge appears in a Q3."""
        edge_counts = defaultdict(int)
        for A, B, cross_edges in splits_data:
            present = 0
            present_edges = []
            for (u, v) in cross_edges:
                if (adj[u] >> v) & 1:
                    present += 1
                    present_edges.append((u, v))
            if present < 12:
                continue
            if present > 12:
                for e in present_edges:
                    edge_counts[e] += 1
                continue
            missing = []
            for (u, v) in cross_edges:
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
            a_seen = set()
            b_seen = set()
            ok = True
            for (u, v) in missing:
                if u in A:
                    if u in a_seen or v in b_seen:
                        ok = False
                        break
                    a_seen.add(u)
                    b_seen.add(v)
                else:
                    if v in a_seen or u in b_seen:
                        ok = False
                        break
                    a_seen.add(v)
                    b_seen.add(u)
            if ok:
                for e in present_edges:
                    edge_counts[e] += 1
        return edge_counts

    random.seed(42)

    current_edges = set()
    current_adj = [0] * n

    # Initial random dense graph
    for e in all_edges:
        if random.random() < 0.9:
            current_edges.add(e)
            u, v = e
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)

    # Repair
    while time.time() < deadline - 25:
        e_to_remove = find_q3_edge_to_remove(current_adj)
        if e_to_remove is None:
            break
        current_edges.remove(e_to_remove)
        u, v = e_to_remove
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)

    best_edges = set(current_edges)
    best_count = len(current_edges)

    # ILS with greedy add
    while time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            non_edges = [e for e in all_edges if e not in current_edges]
            random.shuffle(non_edges)
            for (u, v) in non_edges:
                e = (min(u, v), max(u, v))
                if e in current_edges:
                    continue
                current_edges.add(e)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                if not check_q3_exists(current_adj):
                    improved = True
                    break
                else:
                    current_edges.remove(e)
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)

        if time.time() > deadline - 10:
            break

        num_removals = random.randint(2, 3)
        edges_list = list(current_edges)
        edges_to_remove = random.sample(edges_list, min(num_removals, len(edges_list)))
        for e in edges_to_remove:
            current_edges.remove(e)
            u, v = e
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)

    # Final polish
    if best_edges and time.time() < deadline - 5:
        adj = [0] * n
        for (u, v) in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        added_any = True
        while added_any and time.time() < deadline - 2:
            added_any = False
            non_edges = [e for e in all_edges if e not in best_edges]
            random.shuffle(non_edges)
            for (u, v) in non_edges:
                e = (min(u, v), max(u, v))
                if e not in best_edges:
                    best_edges.add(e)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not check_q3_exists(adj):
                        added_any = True
                        break
                    else:
                        best_edges.remove(e)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

    return [list(e) for e in best_edges]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
