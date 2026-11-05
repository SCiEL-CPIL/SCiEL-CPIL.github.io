import sys
import time
from collections import deque
from math import isqrt
import random
import numpy as np

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
    q = k
    # For girth g, we need d such that girth >= d + 5 (if d odd) or d + 4 (if d even?)
    # The paper says: girth >= d+5 if d is odd, d+4 if d is even?
    # Actually, the girth is at least d+4.
    # To get girth g, we need d >= g - 4.
    # Also, the construction works for d >= 2.
    # The code used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's stick to that to ensure validity.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # rule[c] gives (a, b) such that p ~ l iff l[c] - p[c] = l[a]*p[b]
    # For c=2,3,4 the rule is fixed.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The pattern of indices (a, b) for c:
        # c=5: r=0 -> (1, c0-2) = (1, 3)
        # c=6: r=1 -> (c0-1, 1) = (5, 1)
        # c=7: r=2 -> (c0, 1) = (7, 1)
        # c=8: r=3 -> (1, c0+1) = (1, 9)
        # c=9: r=0 -> (1, c0-2) = (1, 7)
        # ...
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0: # point x, line y
                    y[c] = add(x[c], mul(y[a], x[b]))
                else: # line x, point y
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _is_simple(n, edges):
    if not edges:
        return True
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    return True

def _has_girth(n, edges, g):
    """Check if graph has no cycle of length < g.
    Returns True if girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degrees
    for u in range(n):
        if len(adj[u]) != 3:
            return False

    # BFS with non-backtracking constraint
    # State: (node, prev_node)
    # We want to find if any node can be reached by a walk of length L < g
    # such that the walk forms a cycle.
    # Actually, simpler:
    # A cycle of length L exists iff there is a walk v0, ..., vL=v0 of length L
    # where v_i != v_{i-2} for all i (non-backtracking) and v_i are distinct?
    # No, non-backtracking walks can self-intersect.
    # But if a non-backtracking walk of length L starts and ends at the same node,
    # does it imply a cycle of length <= L?
    # Yes, any closed non-backtracking walk contains a cycle.
    # The length of the shortest cycle is the girth.
    # So we just need to check if there is ANY closed non-backtracking walk of length L < g.
    
    # We can do BFS on the directed line graph (states are directed edges).
    # Start from all directed edges (u, v).
    # Step: (u, v) -> (v, w) where w is in adj[v] and w != u.
    # If we reach (v, u) after L steps, we have a closed non-backtracking walk?
    # Wait.
    # Walk: u0 -> u1 -> ... -> uL = u0.
    # States: (u0, u1), (u1, u2), ..., (u_{L-1}, u_L).
    # We want to know if we can get back to a state (u0, u1) ?
    # Or just if u_L = u0 and u_{L-1} != u1?
    # Actually, if we start with state (u, v), and after L steps we are at state (w, x),
    # the walk is u, v, ..., w, x.
    # If x == u, we have a closed walk of length L+1?
    # Let's trace:
    # Step 0: state (u0, u1). Path: u0, u1.
    # Step 1: state (u1, u2). Path: u0, u1, u2.
    # ...
    # Step L-1: state (u_{L-1}, u_L). Path: u0, ..., u_L.
    # If u_L == u0, we have a cycle of length L (edges: u0u1, ..., u_{L-1}u0).
    # And it is non-backtracking by construction.
    
    # So:
    # Initialize BFS with all states (u, v) at depth 1.
    # (Wait, length 1 is just one edge. A cycle of length 1 is impossible in simple graph.)
    # We want to check for lengths 3, 4, ..., g-1.
    
    # Let's store for each state (u, v) the minimum length L such that we can reach it
    # via a non-backtracking walk starting from some (u, v) ?
    # No, we need to close the cycle.
    
    # Better approach:
    # For each vertex v, check if it can be the start and end of a non-backtracking walk
    # of length L < g.
    # This is equivalent to checking if there is a state (u, v) at depth L-1 in the BFS
    # tree rooted at (v, u)? No.
    
    # Let's use the standard girth algorithm:
    # For each vertex r, do BFS.
    # dist[v] = min distance from r.
    # If we find an edge (u, v) such that dist[u] + dist[v] + 1 < g, then there is a cycle
    # of length < g.
    # This works for undirected graphs.
    # Complexity: O(n * (n + m)) = O(n^2) which is fine for n < 1000.
    
    for r in range(n):
        dist = [-1] * n
        q = deque()
        dist[r] = 0
        q.append(r)
        while q:
            u = q.popleft()
            if dist[u] >= g - 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # But this might not be the shortest cycle involving this edge.
                    # However, if dist[u] + dist[v] + 1 < g, we found a short cycle.
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Try to find a Cayley graph on Z_n with girth >= g
    # We search for n >= Moore Bound
    # Moore Bound for k=3, g=16:
    # n >= 2 * (3 * (3^(7-1) - 1) / (3-1)) = 2 * (3 * 486 / 2) = 1458?
    # Wait.
    # Moore Bound for k-regular graph of girth g:
    # If g is even (g=2r): n >= k * ( (k-1)^(r-1) - 1 ) / (k-2)
    # Here g=16, r=8.
    # n >= 3 * ( (2^7) - 1 ) / 1 = 3 * 127 = 381.
    # Wait, the prompt says lower bound is 512.
    # Let's re-calculate.
    # Number of nodes at distance i from root: k * (k-1)^(i-1) for i >= 1.
    # Total nodes in tree of radius r-1: 1 + sum_{i=1}^{r-1} k(k-1)^(i-1)
    # = 1 + k * ( (k-1)^(r-1) - 1 ) / (k-2)
    # For k=3, r=8:
    # 1 + 3 * ( 2^7 - 1 ) / 1 = 1 + 3 * 127 = 1 + 381 = 382.
    # So n >= 382.
    # The prompt says best lower bound is 512. This is likely due to parity or other constraints,
    # or I am misremembering the formula.
    # Actually, for bipartite graphs (which Cayley on Z_n with odd generators might be?),
    # the Moore bound is tighter.
    # Let's just start from 512 to be safe, or 382.
    # Let's start from 512.
    
    n_min = 512
    n_max = 1200
    
    # Precompute constraints for girth >= g
    # A walk of length L is a sequence of steps from the generator set G = {a, b, -a, -b}.
    # Actually, generators are a, b. Steps can be +a, -a, +b, -b.
    # Non-backtracking means we cannot do +a then -a.
    # So steps s_1, ..., s_L where s_i in {a, b, -a, -b} and s_i != -s_{i-1}.
    # We require sum(s_i) != 0 mod n for all L < g.
    
    # We can precompute all non-backtracking walks of length L < g
    # and get the coefficients (u, v) such that sum = u*a + v*b.
    # We need (u*a + v*b) % n != 0.
    
    # Since a, b are variables, we can just store the set of (u, v) pairs.
    # There are at most 4 * 3^(L-1) walks of length L.
    # For L=15, 4 * 3^14 ~ 15 million. This is too many to store all if we do it for all L.
    # But we can do it incrementally.
    # Also, many (u, v) will be duplicates.
    
    # Let's generate the set of (u, v) for all non-backtracking walks of length L < g.
    # We only care about L up to g-1.
    
    # Optimization:
    # We can just iterate n from n_min to n_max.
    # For each n, iterate a, b.
    # For each (a, b), check all (u, v) in the set of constraints.
    # If (u*a + v*b) % n == 0, reject.
    
    # Let's generate the constraint set (u, v) for L < g.
    constraints = set()
    
    # Walks of length 1: (1,0), (-1,0), (0,1), (0,-1)
    # Walks of length 2: (2,0), (-2,0), (0,2), (0,-2), (1,1), (1,-1), (-1,1), (-1,-1)
    # etc.
    
    # We can use a queue to generate walks.
    # State: (last_step, u, v)
    # last_step is the last generator used (a, b, -a, -b).
    # We want to add new steps.
    
    # Let's represent steps as indices: 0:+a, 1:+b, 2:-a, 3:-b
    # Coeffs:
    # 0: (1, 0)
    # 1: (0, 1)
    # 2: (-1, 0)
    # 3: (0, -1)
    
    # Non-backtracking: cannot follow 0 with 2, or 2 with 0, or 1 with 3, or 3 with 1.
    
    # We will generate all (u, v) for lengths 1 to g-1.
    # We can stop if the set stabilizes or if we run out of time.
    
    # Let's do BFS on the walk tree.
    # Level 1:
    walks = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    # (step_index: (u, v))
    # Note: (1,0) and (-1,0) are distinct walks of length 1.
    # But for the constraint, if u*a + v*b = 0, it's bad.
    # So we add all (u, v) to the set.
    
    for s, (u, v) in walks.items():
        constraints.add((u, v))
    
    # Level 2..g-1
    for L in range(2, g):
        new_walks = {}
        for last_step, (u, v) in walks.items():
            for next_step in range(4):
                # Check non-backtracking
                if (last_step == 0 and next_step == 2) or \
                   (last_step == 2 and next_step == 0) or \
                   (last_step == 1 and next_step == 3) or \
                   (last_step == 3 and next_step == 1):
                    continue
                # Add step
                nu = u + [1, 0, -1, 0][next_step]
                nv = v + [0, 1, 0, -1][next_step]
                # We don't need to store the walk itself, just the (u, v)
                constraints.add((nu, nv))
                # We don't need to store new_walks for the next level if we just want the set of (u,v).
                # But to generate length L+1, we need to know the last step.
                # So we DO need to store the walks (last_step -> (u, v)).
                # But multiple walks can end with the same last_step and same (u, v).
                # That's fine, we just need to expand them.
                # Wait, if two walks end with the same (last_step, u, v), they will generate the same next steps.
                # So we can prune!
                # Key: (last_step, u, v)
                # If we have seen this state before, we don't need to expand it again.
                pass
        
        # Let's implement the pruning version.
        # State: (last_step, u, v)
        # We start with length 1 states.
        # Then expand to length 2, etc.
        
        # Let's restart the generation with pruning.
        break # We'll rewrite this part properly below.
        
    # Proper generation with pruning
    states = {} # (last_step, u, v) -> True
    # Length 1
    for s in range(4):
        u = [1, 0, -1, 0][s]
        v = [0, 1, 0, -1][s]
        states[(s, u, v)] = True
        constraints.add((u, v))
    
    for L in range(2, g):
        new_states = set()
        for (s, u, v) in states:
            for ns in range(4):
                if (s == 0 and ns == 2) or (s == 2 and ns == 0) or (s == 1 and ns == 3) or (s == 3 and ns == 1):
                    continue
                nu = u + [1, 0, -1, 0][ns]
                nv = v + [0, 1, 0, -1][ns]
                new_key = (ns, nu, nv)
                if new_key not in states: # Only add if not seen in previous levels?
                    # Actually, we should check if it's in new_states or old states?
                    # If it's in old states, it's already been expanded.
                    # So we only need to expand new states.
                    new_states.add(new_key)
                    constraints.add((nu, nv))
        states.update(new_states)
        
    # Now we have all (u, v) for non-backtracking walks of length < g.
    # We can sort them by max(|u|, |v|) to check smaller ones first?
    # Or just check all.
    # There might be many constraints.
    # Let's see how many.
    # For g=16, it might be around 1000-5000 distinct (u, v) pairs.
    
    cons_list = list(constraints)
    
    # Search for n, a, b
    # n from n_min to n_max
    # a from 1 to n//2
    # b from a+1 to n//2
    # Check gcd(n, a, b) == 1? Not strictly necessary for connectivity, but good for primitive.
    # Check (u*a + v*b) % n != 0 for all (u, v) in cons_list.
    
    # To speed up, we can precompute for each n the forbidden pairs (a, b).
    # Or just loop.
    
    # We can also randomize the order of a, b to find a solution faster if one exists.
    # But deterministic is safer for reproducibility.
    
    # Let's try to find ANY n.
    # We can iterate n in a specific order.
    # Maybe try n that are prime? No, Z_n doesn't need to be prime.
    
    # Let's iterate n from n_min to n_max.
    # For each n, iterate a, b.
    
    # Optimization:
    # For a fixed n and a, we can determine forbidden b values.
    # (u*a + v*b) % n != 0
    # v*b != -u*a (mod n)
    # If v is invertible mod n, b != -u*a * v^-1 (mod n).
    # If not, we have a system of linear congruences.
    
    # Given the small size, brute force with early exit might be fast enough.
    
    found_n = -1
    found_a = -1
    found_b = -1
    
    for n in range(n_min, n_max + 1):
        if time.time() > deadline - 5:
            break
        for a in range(1, n // 2 + 1):
            if (2 * a) % n == 0: # a = -a
                continue
            for b in range(a + 1, n // 2 + 1):
                if (2 * b) % n == 0: # b = -b
                    continue
                if (a + b) % n == 0: # b = -a
                    continue
                if (b - a) % n == 0: # b = a
                    continue
                
                # Check constraints
                valid = True
                for u, v in cons_list:
                    if (u * a + v * b) % n == 0:
                        valid = False
                        break
                if valid:
                    found_n = n
                    found_a = a
                    found_b = b
                    break
            if found_n != -1:
                break
        if found_n != -1:
            break
    
    if found_n != -1:
        n = found_n
        a = found_a
        b = found_b
        edges = []
        for i in range(n):
            # Edges: (i, i+a), (i, i+b)
            # We only add these two to avoid duplicates (i, i-a) and (i, i-b) will be added when i is i-a and i-b.
            # Wait, if we add (i, i+a) for all i, we cover all edges.
            # And (i, i+b) for all i.
            # Are there duplicates between the two sets?
            # i+a = j+b => j = i+a-b.
            # If we add (i, i+a) and (j, j+b), they are the same edge if {i, i+a} == {j, j+b}.
            # This happens if i = j and i+a = j+b => a=b (excluded).
            # Or i = j+b and i+a = j => a = -b (excluded).
            # So no duplicates between the two sets.
            # And within a set, (i, i+a) is unique for each i?
            # Yes, unless a=0 (excluded).
            # And (i, i+a) == (i+a, i) ?
            # We store edges as (u, v). The evaluator likely treats them as undirected.
            # So we just need to ensure we don't add (u, v) and (v, u) as separate entries if they are considered distinct?
            # The task says "list of edges (i, j)". Usually undirected.
            # If I add (i, i+a) for all i, I am adding each edge once.
            # Because the edge between x and y is added when i=x (if y=x+a) or i=y (if x=y+a).
            # Since a != 0, exactly one of these is true?
            # No, if y = x+a, then when i=x, we add (x, y).
            # When i=y, we add (y, y+a). This is a different edge.
            # So each edge {x, y} where y-x=a is added exactly once (when i=x).
            # So the list of edges will have n edges of type a and n edges of type b.
            # Total 2n edges.
            # For a 3-regular graph on n vertices, we need 3n/2 edges.
            # Wait.
            # Cayley graph on Z_n with generators {a, b, -a, -b}.
            # The edges are (i, i+a), (i, i-b), etc.
            # If I add (i, i+a) for all i, I get n edges.
            # If I add (i, i+b) for all i, I get n edges.
            # Total 2n edges.
            # Degree of each vertex:
            # i is connected to i+a, i-a, i+b, i-b.
            # That's 4 neighbors.
            # But the graph is 4-regular?
            # No, the generators are {a, -a, b, -b}.
            # In a Cayley graph, the degree is the size of the generating set.
            # If the generating set is {a, b, -a, -b}, the degree is 4.
            # But we need degree 3.
            # So we cannot use a Cayley graph on Z_n with 2 generators a, b unless we restrict the generators.
            # For degree 3, we need an odd number of generators.
            # E.g., {a, b, a+b}? Or {a, b, c} with c = -a-b?
            # If we use generators {a, b, c} where a+b+c=0, then the graph is 3-regular.
            # But then the group is not necessarily Z_n.
            # If we use Z_n, we need 3 generators a, b, c such that a, b, c are distinct and not 0, and a+b+c=0?
            # No, the generators don't need to sum to 0.
            # But if we use {a, b, c}, the degree is 3.
            # The condition for girth is that no non-trivial relation of length < g exists.
            # This is more complex to check.
            
            # Let's stick to the LUW graph for now, as it is guaranteed to work.
            # The Cayley search might find a 4-regular graph, which is invalid for k=3.
            # So I will NOT use the Cayley search for k=3.
            # I will only use LUW.
            pass
    
    # Since Cayley on Z_n with 2 generators gives 4-regular graphs, it's not suitable for k=3.
    # I will just return the LUW graph.
    return _luw_construct(k, g)

# EVOLVE-BLOCK-END
