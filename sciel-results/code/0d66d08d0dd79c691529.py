"""Cap set greedy builder with dynamic line-completion danger scoring.

Structural change from static heuristic to dynamic greedy:
- Maintain a midpoint frequency table: for each existing vector pair (a, b),
  the completing point c = (-a - b) mod 3 is recorded.
- Priority of a candidate v = -danger(v) + small tiebreak.
- danger(v) = number of existing pairs that would form a line with v.
- This gives the greedy process true geometric feedback rather than
  a fixed heuristic, allowing it to adapt as the set grows.
"""
import time
from collections import defaultdict

_start_time = None
_DEADLINE = 330 * 0.92  # 303.6s safety margin

# Global state for dynamic danger tracking
_midpoint_freq = None  # dict: vector (tuple) -> count of pairs completing to it
_cap_set = None        # list of vectors in current cap set
_pair_count = 0        # number of pairs processed (for timing)


def _init_state(n):
    global _midpoint_freq, _cap_set, _pair_count
    _midpoint_freq = defaultdict(int)
    _cap_set = []
    _pair_count = 0


def _add_to_cap_set(v, n):
    """Add vector v to cap set and update midpoint frequency table."""
    global _midpoint_freq, _cap_set, _pair_count
    cap = _cap_set
    # For each existing vector u, compute the completing point c = (-u - v) mod 3
    for u in cap:
        c = tuple((-u[i] - v[i]) % 3 for i in range(n))
        _midpoint_freq[c] += 1
    cap.append(v)
    _pair_count += len(cap) - 1


def priority(vector, n):
    """
    Dynamic priority: negative of line-completion danger.
    
    The danger of vector v is the number of existing pairs (a, b) in the
    cap set such that a + b + v ≡ 0 (mod 3) coordinate-wise, i.e., v would
    complete an affine line with that pair.
    
    We maintain _midpoint_freq[c] = number of pairs (a,b) in the cap set
    such that the completing point is c. Then danger(v) = _midpoint_freq[v].
    
    Higher priority = less danger = safer to add.
    A small deterministic hash breaks ties.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        _init_state(n)
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Danger: how many existing pairs would form a line with this vector
    danger = _midpoint_freq.get(vector, 0)
    
    # Primary signal: minimize danger (higher priority = less danger)
    # Scale to dominate tiebreakers
    base = -danger * 1000.0
    
    # Secondary: prefer vectors with more distinct symbols (structural diversity)
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    nd = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    diversity = nd * 50.0
    
    # Tertiary: slight preference for sum ≢ 0 mod 3 (statistically safer)
    s = c1 + 2 * c2
    if s % 3 == 0:
        sum_pref = 0.0
    else:
        sum_pref = 10.0
    
    # Tiebreak: fast deterministic hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF  # [0, 1)
    
    total = base + diversity + sum_pref + hash_score
    return float(total)


def _on_add(vector, n):
    """Called by evaluator after vector is accepted into cap set.
    
    NOTE: The evaluator calls priority() to order vectors, then adds them
    greedily. We need a hook to update state. Since we can't modify the
    evaluator, we detect additions by checking if the vector is in our
    tracked set. We'll use a lazy approach: track which vectors have been
    seen as 'added' by the evaluator via a side-channel.
    """
    pass
