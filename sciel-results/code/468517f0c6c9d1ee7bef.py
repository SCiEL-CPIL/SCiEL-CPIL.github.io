"""Hierarchical decomposition: fix a ping-pong core, optimize remaining transitions via beam search."""
import time
import random
from collections import deque

N = 12

def _compute_reset_length(trans):
    """BFS on subset automaton to find shortest reset word length.
    Returns (length, word) or (None, None) if not synchronizing."""
    # trans: list of two lists, trans[0][i] = delta(i, 'a'), trans[1][i] = delta(i, 'b')
    # Full set is all states
    full = (1 << N) - 1
    
    # BFS from full set
    # State: subset as bitmask
    # Precompute transitions for all subsets? No, that's 2^12 = 4096 subsets, each with 2 transitions.
    # Let's precompute the transition table for all subsets.
    
    # trans_table[subset][letter] = next_subset
    # subset is an integer bitmask
    # For efficiency, precompute for all subsets
    
    # Actually, let's just do BFS on the fly but cache transitions
    trans_cache = {}
    
    def get_next_subset(subset, letter):
        key = (subset, letter)
        if key in trans_cache:
            return trans_cache[key]
        next_subset = 0
        for s in range(N):
            if (subset >> s) & 1:
                next_state = trans[letter][s]
                next_subset |= (1 << next_state)
        trans_cache[key] = next_subset
        return next_subset
    
    # BFS
    if full == 0:
        return (0, "")
    
    # Check if full is already a singleton (not possible for N>1)
    if full & (full - 1) == 0:
        return (0, "")
    
    visited = set()
    visited.add(full)
    queue = deque()
    queue.append((full, ""))
    
    while queue:
        subset, word = queue.popleft()
        for letter in range(2):
            next_subset = get_next_subset(subset, letter)
            if next_subset not in visited:
                if next_subset & (next_subset - 1) == 0:
                    # Found a singleton
                    return (len(word) + 1, word + str(letter))
                visited.add(next_subset)
                queue.append((next_subset, word + str(letter)))
    
    return (None, None)

def _is_synchronizing(trans):
    length, _ = _compute_reset_length(trans)
    return length is not None

def construct():
    """Construct a 12-state binary automaton with long reset threshold.
    
    Strategy: Start with a known high-threshold structure and optimize via beam search.
    """
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for the 90s limit
    
    # Initial structure: ping-pong cycle
    # States 0-10 form a cycle: 0->1->2->...->10->0 on 'a'
    # State 11 is a "sink" that maps to 0 on 'a'
    # On 'b', we'll optimize
    
    # Let's start with a simple structure and optimize
    # trans[0] = 'a' transitions, trans[1] = 'b' transitions
    
    # Initial guess: 'a' is a cycle, 'b' maps everything to a small set
    best_trans = None
    best_length = 0
    
    # Try several initial structures
    initial_structures = []
    
    # Structure 1: 'a' is a full cycle, 'b' maps all to state 0
    a_trans1 = [(i + 1) % N for i in range(N)]
    b_trans1 = [0] * N
    initial_structures.append(([a_trans1, b_trans1], "cycle_to_zero"))
    
    # Structure 2: 'a' is a full cycle, 'b' maps to alternating states
    a_trans2 = [(i + 1) % N for i in range(N)]
    b_trans2 = [i % 2 for i in range(N)]
    initial_structures.append(([a_trans2, b_trans2], "cycle_alternating"))
    
    # Structure 3: 'a' is a full cycle, 'b' maps to a random-looking pattern
    a_trans3 = [(i + 1) % N for i in range(N)]
    b_trans3 = [(i * 5) % N for i in range(N)]
    initial_structures.append(([a_trans3, b_trans3], "cycle_linear"))
    
    # Structure 4: 'a' is a full cycle, 'b' maps to a permutation
    a_trans4 = [(i + 1) % N for i in range(N)]
    b_trans4 = [(i * 7) % N for i in range(N)]
    initial_structures.append(([a_trans4, b_trans4], "cycle_perm"))
    
    # Structure 5: 'a' is a full cycle, 'b' maps to a specific pattern that creates long resets
    # Based on known Černý constructions
    a_trans5 = [(i + 1) % N for i in range(N)]
    b_trans5 = [0] * N
    # Make b map some states differently to create complexity
    for i in range(N):
        if i % 3 == 0:
            b_trans5[i] = (i + 3) % N
    initial_structures.append(([a_trans5, b_trans5], "cycle_mod3"))
    
    # Evaluate initial structures
    for trans, name in initial_structures:
        length, _ = _compute_reset_length(trans)
        if length is not None and length > best_length:
            best_length = length
            best_trans = [trans[0][:], trans[1][:]]
    
    # Beam search to improve
    beam_width = 5
    beam = []
    
    # Add the best initial structure to the beam
    if best_trans is not None:
        beam.append((best_trans[0][:], best_trans[1][:], best_length))
    
    # Also add some random structures to the beam
    for _ in range(beam_width - len(beam)):
        a_rand = [(i + 1) % N for i in range(N)]  # Keep 'a' as a cycle
        b_rand = [random.randint(0, N-1) for _ in range(N)]
        trans = ([a_rand, b_rand])
        length, _ = _compute_reset_length(trans)
        if length is not None:
            beam.append((a_rand[:], b_rand[:], length))
    
    # Sort beam by length (descending)
    beam.sort(key=lambda x: x[2], reverse=True)
    beam = beam[:beam_width]
    
    # Beam search iterations
    max_iterations = 1000
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        
        new_beam = []
        
        for a_trans, b_trans, length in beam:
            # Generate neighbors by changing one transition
            for _ in range(10):  # Generate 10 neighbors per beam member
                neighbor_a = a_trans[:]
                neighbor_b = b_trans[:]
                
                # Choose which transition to change
                if random.random() < 0.5:
                    # Change 'a' transition
                    state = random.randint(0, N-1)
                    neighbor_a[state] = random.randint(0, N-1)
                else:
                    # Change 'b' transition
                    state = random.randint(0, N-1)
                    neighbor_b[state] = random.randint(0, N-1)
                
                trans = ([neighbor_a, neighbor_b])
                new_length, _ = _compute_reset_length(trans)
                
                if new_length is not None:
                    new_beam.append((neighbor_a[:], neighbor_b[:], new_length))
        
        # Keep the best beam_width candidates
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
        
        # Update best
        if beam and beam[0][2] > best_length:
            best_length = beam[0][2]
            best_trans = [beam[0][0][:], beam[0][1][:]]
        
        # If no improving candidates, break
        if not beam:
            break
    
    if best_trans is None:
        # Fallback to the simplest synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_trans[0][:], best_trans[1][:]]

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
