import time
import itertools
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples as bitmasks (3 bits set)
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))
    
    # Also build for each node, the list of triples it participates in
    triples_by_node = [[] for _ in range(n)]
    for idx, mask in enumerate(bad_triples):
        for bit in range(n):
            if (mask >> bit) & 1:
                triples_by_node[bit].append(idx)
    
    # Fast check: is a triple satisfied (all 3 present)?
    def is_satisfied(mask, triple_mask):
        return (mask & triple_mask) == triple_mask
    
    # Count violations for a given assignment mask
    def count_violations(assignment):
        cnt = 0
        for tm in bad_triples:
            if (assignment & tm) == tm:
                cnt += 1
        return cnt
    
    # Greedy initialization: start with all nodes, remove to break all bad triples
    # This is a hitting set / vertex cover on 3-uniform hypergraph
    def greedy_init():
        # Start with all selected
        selected = (1 << n) - 1
        # Randomize removal order slightly for diversity
        order = list(range(n))
        random.shuffle(order)
        for node in order:
            # Check if removing this node breaks any currently-satisfied triple
            # Actually, let's do: start empty, add nodes greedily
            pass
        
        # Better: start with all, iteratively remove nodes that are in most violations
        selected = (1 << n) - 1
        changed = True
        while changed:
            if time.time() > deadline - 60:
                break
            changed = False
            # Find most "offending" node
            best_node = -1
            best_count = 0
            # Only consider nodes in selected
            temp = selected
            while temp:
                lsb = temp & (-temp)
                node = lsb.bit_length() - 1
                cnt = 0
                for tidx in triples_by_node[node]:
                    tm = bad_triples[tidx]
                    if (selected & tm) == tm:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_node = node
                temp ^= lsb
            
            if best_node != -1 and best_count > 0:
                selected &= ~(1 << best_node)
                changed = True
        
        return selected
    
    # WalkSAT-style local search
    def walk_sat(initial_mask, num_iters=500000):
        assignment = initial_mask
        current_violations = set()
        for tm in bad_triples:
            if (assignment & tm) == tm:
                current_violations.add(tm)
        
        best_assignment = assignment
        best_size = bin(assignment).count('1')
        
        for it in range(num_iters):
            if time.time() > deadline - 10:
                break
            
            # Try to add a random unselected node if it doesn't create a violation
            # Or flip a node
            if random.random() < 0.3:
                # Try adding a random unselected node
                unselected = ((1 << n) - 1) & ~assignment
                if unselected:
                    lsb = unselected & (-unselected)
                    node = lsb.bit_length() - 1
                    # Check if adding node creates a violation
                    new_violations = 0
                    for tidx in triples_by_node[node]:
                        tm = bad_triples[tidx]
                        # Check if the other two are in assignment
                        others = tm & ~(1 << node)
                        if (assignment & others) == others:
                            new_violations += 1
                    if new_violations == 0:
                        assignment |= (1 << node)
                        # Update violations (none added)
                        sz = bin(assignment).count('1')
                        if sz > best_size:
                            best_size = sz
                            best_assignment = assignment
                        continue
            
            # Flip a random node
            lsb = assignment & (-assignment) if assignment else 0
            if not lsb:
                # All zero, pick random
                node = random.randint(0, n-1)
                lsb = 1 << node
            else:
                # Weighted: prefer nodes in violations
                violating_nodes = set()
                for vm in current_violations:
                    for bit in range(n):
                        if (vm >> bit) & 1:
                            violating_nodes.add(bit)
                
                if violating_nodes and random.random() < 0.7:
                    node = random.choice(list(violating_nodes))
                else:
                    node = random.randint(0, n-1)
                lsb = 1 << node
            
            # Evaluate flip
            if (assignment >> node) & 1:
                # Remove node: count how many violations are broken
                broken = 0
                for tidx in triples_by_node[node]:
                    tm = bad_triples[tidx]
                    if tm in current_violations:
                        broken += 1
                # Can't easily count created violations when removing (none)
                # Accept if broken > 0 or random
                if broken > 0 or random.random() < 0.1:
                    assignment &= ~lsb
                    # Update violations
                    new_violations = set()
                    for vm in current_violations:
                        if (vm >> node) & 1:
                            # This violation might be broken
                            if (assignment & vm) != vm:
                                pass  # broken
                            else:
                                new_violations.add(vm)
                        else:
                            new_violations.add(vm)
                    current_violations = new_violations
                    sz = bin(assignment).count('1')
                    if sz > best_size:
                        best_size = sz
                        best_assignment = assignment
            else:
                # Add node: count how many violations are created
                created = 0
                for tidx in triples_by_node[node]:
                    tm = bad_triples[tidx]
                    others = tm & ~lsb
                    if (assignment & others) == others:
                        created += 1
                # Accept if created == 0 or random
                if created == 0 or random.random() < 0.05:
                    assignment |= lsb
                    # Update violations
                    for tidx in triples_by_node[node]:
                        tm = bad_triples[tidx]
                        if (assignment & tm) == tm:
                            current_violations.add(tm)
                    sz = bin(assignment).count('1')
                    if sz > best_size:
                        best_size = sz
                        best_assignment = assignment
        
        return best_assignment
    
    # Multiple greedy inits with different random seeds
    random.seed(42)
    best_mask = 0
    best_size = 0
    
    # Greedy init
    g_mask = greedy_init()
    g_size = bin(g_mask).count('1')
    if g_size > best_size:
        best_size = g_size
        best_mask = g_mask
    
    # Try all-odd start
    odd_mask = 0
    for i in range(n):
        if (i + 1) % 2 == 1:
            odd_mask |= (1 << i)
    # Check violations in odd set
    odd_viol = 0
    for tm in bad_triples:
        if (odd_mask & tm) == tm:
            odd_viol += 1
    if odd_viol == 0:
        o_size = bin(odd_mask).count('1')
        if o_size > best_size:
            best_size = o_size
            best_mask = odd_mask
    
    # WalkSAT from best init
    if time.time() < deadline - 15:
        ws_mask = walk_sat(best_mask, num_iters=300000)
        ws_size = bin(ws_mask).count('1')
        if ws_size > best_size:
            best_size = ws_size
            best_mask = ws_mask
    
    # Local search: try adding any node that doesn't create a violation
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for node in range(n):
                if not (best_mask >> node) & 1:
                    # Check if adding node creates a violation
                    can_add = True
                    for tidx in triples_by_node[node]:
                        tm = bad_triples[tidx]
                        others = tm & ~(1 << node)
                        if (best_mask & others) == others:
                            can_add = False
                            break
                    if can_add:
                        best_mask |= (1 << node)
                        improved = True
    
    # Final local search: try swapping
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for out_node in range(n):
                if not (best_mask >> out_node) & 1:
                    for in_node in range(n):
                        if (best_mask >> in_node) & 1:
                            # Try swap
                            new_mask = (best_mask & ~(1 << in_node)) | (1 << out_node)
                            # Check violations
                            valid = True
                            for tm in bad_triples:
                                if (new_mask & tm) == tm:
                                    valid = False
                                    break
                            if valid:
                                best_mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
    
    return [i + 1 for i in range(n) if (best_mask >> i) & 1]
