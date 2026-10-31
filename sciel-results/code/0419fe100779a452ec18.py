import time
import sys

# Global state for deadline tracking and dynamic frequency histogram
_start_time = None
_DEADLINE = 300.0
# Histogram of counts for each symbol (0, 1, 2) across all coordinates in the selected set
# Index 0: count of 0s, Index 1: count of 1s, Index 2: count of 2s
_symbol_counts = [0, 0, 0]
_total_coords = 0

def priority(vector: tuple, n: int) -> float:
    """
    Strategy: Dynamic Symbol Frequency Balance Control
    
    This approach treats the greedy selection as a control problem where we want to
    maintain the symbol distribution (0s, 1s, 2s) as close to uniform (1/3, 1/3, 1/3)
    as possible. 
    
    The intuition is that cap sets in $\mathbb{Z}_3^n$ are maximized when the set is
    "balanced" in terms of symbol usage. A greedy algorithm that always picks the 
    highest priority vector might accidentally favor one symbol over others if the 
    priority function has a static bias (like sum of squares). By dynamically 
    penalizing vectors that increase the frequency of an already over-represented 
    symbol, we encourage the set to explore the space more evenly, which often 
    correlates with larger independent set sizes in structured graphs/lattices.
    
    The priority is defined as:
    P(v) = - \sum_{i=0}^{2} \alpha_i * (count_i + n_v_i)^2
    
    Where:
    - count_i is the current global count of symbol i.
    - n_v_i is the number of times symbol i appears in the candidate vector v.
    - We minimize the squared deviation from the current state, effectively 
      implementing a "least recent used" or "least frequent symbol" bias.
      
    To make it a "priority" (higher is better), we return the negative of this cost.
    We can also add a small tie-breaker based on the vector's hash to ensure 
    deterministic but varied exploration if scores are close, though the 
    frequency term should dominate.
    """
    global _start_time, _symbol_counts, _total_coords
    
    # Initialize timer on first call
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # If we are close to the deadline, we cannot update state safely or 
        # might be in a tight loop. Return a neutral value.
        # Note: The evaluator calls priority() to determine order, then adds.
        # If we return 0.0, it might break the order if many are 0.0.
        # However, usually the loop finishes well before 300s for n<30.
        return 0.0

    # Count symbols in the candidate vector
    v_counts = [0, 0, 0]
    for v in vector:
        if v == 0:
            v_counts[0] += 1
        elif v == 1:
            v_counts[1] += 1
        elif v == 2:
            v_counts[2] += 1

    # Calculate the "cost" of adding this vector
    # We want to minimize the sum of squares of the future counts.
    # Future count i = current count i + v_counts[i]
    # Cost = sum( (current[i] + v_counts[i])^2 )
    
    # Optimization: 
    # sum( (c_i + v_i)^2 ) = sum( c_i^2 + 2*c_i*v_i + v_i^2 )
    # sum( c_i^2 ) is constant for all candidates in this step.
    # So we just need to minimize: 2 * sum( c_i * v_i ) + sum( v_i^2 )
    
    linear_term = 0
    quad_term = 0
    
    for i in range(3):
        linear_term += _symbol_counts[i] * v_counts[i]
        quad_term += v_counts[i] * v_counts[i]
        
    cost = 2.0 * linear_term + quad_term
    
    # Priority is negative cost (lower cost is better)
    # We can scale it if needed, but relative order is what matters.
    return -float(cost)

# Note: The evaluator does not provide a callback to update the histogram 
# when a vector is actually added to the set. 
# 
# CRITICAL ISSUE: 
# The interface `priority(vector, n)` is called to rank vectors. 
# The evaluator adds them in descending order. 
# We do NOT know which vectors were added until the next call? 
# No, the evaluator typically calls priority for ALL candidates, sorts them, 
# and then iterates. Or it calls priority one by one?
# 
# "vectors of {0,1,2}^n are added greedily in descending priority"
# This implies:
# 1. Generate all vectors? (Too many for n>10)
# 2. Or does the evaluator provide a stream?
# 
# Looking at the seed program, it doesn't maintain state of *which* vectors 
# were added. It just calculates a static property.
# 
# If the evaluator calls `priority` for each vector in a pre-generated list 
# and then sorts, we can't update `_symbol_counts` because we don't know 
# the final order until the sort is done, and we don't get notified of 
# "vector X was added".
#
# HOWEVER, many FunSearch-style evaluators for cap sets work by:
# 1. Providing a fixed set of candidates (or generating them).
# 2. Calling `priority` for each.
# 3. Sorting by priority.
# 4. Iterating through the sorted list, adding to cap set if valid.
#
# If this is the case, my dynamic histogram approach is BROKEN because 
# `_symbol_counts` will never be updated. It will stay [0,0,0].
# Then `cost = quad_term` (sum of squares of symbols in v).
# This is effectively the same as the current "Sum of Squares" strategy 
# (which prioritizes vectors with fewer 2s? No, sum of squares of v_counts).
# v_counts[2] is count of 2s. 2^2=4. So it penalizes 2s.
# The current program returns `sum(v*v)`. Higher is better.
# My program returns `-sum(v_i^2)`. Lower cost (more 0s) is better.
# This is the OPPOSITE of the current program.
#
# Is there a way to make it stateful?
# If the evaluator calls `priority` sequentially in the order of addition, 
# we could update. But "descending priority" implies a batch sort.
#
# Let's re-read the contract: "vectors ... are added greedily in descending priority".
# This usually means:
# candidates = [...]
# candidates.sort(key=priority, reverse=True)
# for v in candidates:
#    if is_cap_set(current_set + v):
#       add(v)
#
# In this standard model, `priority` is a pure function of the vector. 
# Any stateful approach that relies on knowing *what was added* is impossible 
# unless we can infer it or the interface is different.
#
# HOWEVER, the prompt says: "Treat the priority assignment as a dynamic feedback 
# control problem where the heuristic adapts to the current state of the cap set 
# by tracking the marginal 'line-completion risk'..."
#
# If I cannot track the actual set, I can track a *proxy* for the set.
# But if I can't know what was added, I can't track it.
#
# Alternative Interpretation:
# Maybe the evaluator calls `priority` *during* the greedy step?
# i.e.
# best_v = None
# best_p = -inf
# for v in candidates:
#    if valid(set + v):
#       p = priority(v, n)
#       if p > best_p:
#          best_p = p
#          best_v = v
# add(best_v)
#
# If this is the case, `priority` is called only for *valid* candidates at each step.
# And the order of calls might be arbitrary.
# But even then, we don't get told "v was added".
#
# Let's look at Inspiration 1: "uses time; defines priority".
# Inspiration 2: "uses math, numpy, time, zlib".
# Inspiration 3: "uses collections, time".
#
# If the standard model is batch sort, stateful approaches are impossible 
# unless the state is derived solely from the vector itself (static) 
# or from the *order* of calls (which is not the addition order).
#
# Wait, if the evaluator generates vectors in a specific order (e.g., lexicographic) 
# and calls priority, we can't know the addition order.
#
# Let's assume the "Dynamic Feedback" directive implies that we *can* maintain state 
# that correlates with the additions. 
# If we assume the greedy algorithm picks the highest priority, and we return 
# priorities that depend on a global counter that increments on *every* call, 
# that doesn't help.
#
# Let's look at the "line-completion risk" part.
# Risk is determined by the *current set*. We don't have the current set.
#
# Perhaps the "Dynamic Feedback" is a misinterpretation of what is possible 
# under a pure `priority(v, n)` interface in a batch-sort scenario.
# 
# BUT, what if the evaluator *does* allow statefulness by calling priority 
# in a way that lets us update? 
# No, `priority` returns a float. It doesn't take a "set" argument.
#
# Let's reconsider the "Sum of Squares" strategy (current). 
# It scores 1.0.
# Inspiration 1 (1.30) uses "Hamming-weight symmetry-breaking".
# Hamming weight is the count of non-zeros.
# 
# If I cannot be truly dynamic, I should implement a *static* heuristic 
# that *mimics* the effect of balancing.
# 
# A vector with a balanced mix of 0,1,2 is less likely to form lines 
# with other balanced vectors?
# Actually, lines in Z_3^n are of the form {x, x+d, x+2d}.
# If d is constant, the symbols shift.
#
# Let's try a different static metric: **Entropy of the symbol distribution in the vector**.
# Vectors with high entropy (mixed symbols) might be better because they 
# "use up" all symbol slots evenly, preventing the set from becoming 
# skewed towards 0s or 2s which might lead to dense regions.
#
# However, the directive specifically asks for "dynamic feedback control... 
# tracking marginal line-completion risk... maintaining a global histogram".
# 
# If I implement the histogram but it never updates, it's useless.
# 
# Is it possible that the evaluator calls `priority` *after* adding?
# No, "added greedily in descending priority".
#
# Let's look at the code structure of similar tasks. 
# Often, `priority` is called for every vector in a pre-computed list.
# 
# If I *must* follow the directive to use a dynamic histogram, but the interface 
# prevents true dynamism, I have two choices:
# 1. Implement the histogram, assume it stays [0,0,0], and it degrades to a static metric.
# 2. Find a way to make the "state" depend on the *call order* or *vector identity* 
#    in a way that approximates balancing.
#
# Actually, there is a trick. If the evaluator sorts by priority, and I want to 
# balance the set, I can assign priorities based on a *hash* of the vector that 
# correlates with symbol balance.
#
# Let's stick to the directive's *intent*: Balance the symbol distribution.
# Since I can't see the set, I will use a **Static Balanced Entropy** score.
# I will score vectors highly if they have a mix of 0, 1, and 2.
# A vector like (0,1,2,0,1,2) has high entropy. A vector like (0,0,0,0,0,0) has low.
# High entropy vectors are less likely to be part of long arithmetic progressions 
# that are "monochromatic" in symbol distribution? 
# Actually, APs in Z_3^n don't have to be monochromatic.
#
# Let's try **Negative Variance of Symbols**.
# For a vector v, calculate the variance of its elements.
# Low variance (e.g., all 1s) -> Bad?
# High variance (mix of 0 and 2) -> Good?
#
# Let's go with **Symbol Diversity Score**.
# Score = (count_0 * count_1 * count_2) + 0.1 * (count_0 + count_1 + count_2)
# This rewards vectors that contain all three symbols.
#
# BUT, the directive says "dynamic". 
# If I can't be dynamic, I should at least *structure* the code as if I am, 
# or use a state that *does* update: **Time-based decay**? No.
#
# Let's look at Inspiration 2: "Discrete Fourier Transform".
# This is static per vector.
#
# I will implement a **Static Symbol Balance** heuristic, as true dynamic 
# feedback is impossible without a set-state callback. I will frame it as 
# "approximating the asymptotic balanced state".
#
# Heuristic: 
# 1. Count 0s, 1s, 2s in v.
# 2. Calculate deviation from n/3.
# 3. Penalty = sum((count_i - n/3)^2).
# 4. Priority = -Penalty.
#
# This favors vectors where each symbol appears approximately n/3 times.
# This is the "optimal asymptotic cap set ratio" mentioned in the directive.

def priority(vector: tuple, n: int) -> float:
    """
    Strategy: Static Asymptotic Symbol Balance
    
    Since the interface does not allow tracking the actual cap set state 
    during the greedy selection (standard batch-sort evaluation), we approximate 
    the "dynamic feedback" goal by statically favoring vectors that exhibit 
    the symbol distribution of an optimal, large cap set.
    
    In a balanced cap set in Z_3^n, each symbol (0, 1, 2) should appear 
    with frequency close to 1/3 across the coordinates. Vectors that are 
    "balanced" (contain a mix of 0, 1, 2) are more likely to be compatible 
    with a globally balanced set, whereas "skewed" vectors (mostly 0s) 
    might lead to local density issues or early termination of the greedy 
    process in one corner of the space.
    
    We define the priority as the negative squared deviation of the vector's 
    symbol counts from the ideal uniform distribution (n/3).
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        return 0.0

    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
        
    ideal = n / 3.0
    
    penalty = 0.0
    for i in range(3):
        diff = counts[i] - ideal
        penalty += diff * diff
        
    # Higher priority for lower penalty
    return -float(penalty)
