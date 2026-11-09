# EVOLVE-BLOCK-START
import time

def priority(vector, n):
    # Dynamic greedy strategy:
    # 1. Prioritize vectors with 'balanced' digit distribution (more 1s is often better for cap sets in higher dimensions, 
    #    as {0,2}^n is a lower bound. Vectors with many 1s can extend the set).
    # 2. Penalize vectors that would form arithmetic progressions with existing set elements.
    #    However, the interface `priority(vector, n)` does NOT have access to the current set of selected vectors.
    #    Wait, the task says: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set".
    #    This implies the evaluator calls `priority` for a candidate vector, and if it's valid (keeps cap set property), it's added.
    #    Crucially, `priority` is called for *each* candidate vector in some order? Or does it return a score that determines the order?
    #    "added greedily in descending priority" suggests we sort all 3^n vectors by priority, then iterate, adding if valid.
    #    If we sort all vectors upfront, we don't know the current set state during sorting.
    #    BUT, if the evaluator calls `priority` repeatedly for the "next best" candidate, or if it just uses the return value to sort once,
    #    we must assume the latter unless the interface supports state.
    #    The signature is `priority(vector, n) -> float`. It has no state.
    #    Therefore, we cannot know which vectors are already in the set when computing the priority for a new one, UNLESS the evaluator
    #    calls priority dynamically. But "greedily in descending priority" usually implies a pre-computed order or a static score.
    #    
    #    Let's re-read carefully: "vectors ... are added greedily in descending priority when they keep the set a cap set".
    #    This phrasing often implies:
    #    1. Compute priority for all vectors.
    #    2. Sort them.
    #    3. Iterate through sorted list, add if valid.
    #    
    #    If this is the case, a static heuristic is the only option.
    #    The current program uses a static heuristic: prefer {0,2}^n.
    #    The inspiration says: "Balanced digit distribution heuristic ... Measured: 1 -> 1.223".
    #    This suggests that prioritizing vectors with a mix of 0, 1, 2 (balanced) yields a larger cap set than just {0,2}^n.
    #    
    #    Why? {0,2}^n is a cap set of size 2^n. The maximum cap set size in {0,1,2}^n is known to be larger than 2^n for n>=3.
    #    For example, in n=3, max cap set is 9, while 2^3=8.
    #    A vector like (1,1,1) is not in {0,2}^3. If we prioritize it, we might get a larger set.
    #    
    #    What is a good static heuristic for "balanced"?
    #    Vectors with digits close to the mean (1) are "balanced".
    #    We can score based on the number of 1s.
    #    Or based on the variance of the digits.
    #    Let's try prioritizing vectors with more 1s.
    #    Current: if any 1, return 0.0. Else 1.0.
    #    New: return count of 1s.
    #    This will prioritize (1,1,...,1) first, then (0,1,...,1), etc.
    #    
    #    Let's test this hypothesis mentally.
    #    If we pick (1,1,1) first. Then (0,0,0) is valid? (1,1,1) and (0,0,0) and (2,2,2) form an AP?
    #    0, 1, 2 is an AP. So (0,0,0), (1,1,1), (2,2,2) is an AP.
    #    So if we pick (1,1,1), we cannot pick both (0,0,0) and (2,2,2).
    #    
    #    The {0,2}^n strategy picks all vectors with no 1s.
    #    (0,0,0) and (2,2,2) are both in {0,2}^n. They do NOT form an AP with each other alone.
    #    Wait, an AP in {0,1,2}^n is a set {a, b, c} such that a+c = 2b.
    #    In {0,2}^n, if a and c are in {0,2}^n, then 2b = a+c.
    #    If a_i, c_i in {0,2}, then (a_i+c_i) is in {0,2,4}.
    #    2b_i must be in {0,2,4}. So b_i must be in {0,1,2}.
    #    If b is also in {0,2}^n, then b_i in {0,2}.
    #    If a_i=0, c_i=0, 2b_i=0 => b_i=0.
    #    If a_i=0, c_i=2, 2b_i=2 => b_i=1. But b_i must be 0 or 2. Contradiction.
    #    So no AP of length 3 exists entirely within {0,2}^n.
    #    Thus {0,2}^n is always a cap set.
    #    
    #    Can we do better?
    #    Yes, by adding some vectors with 1s.
    #    If we prioritize vectors with 1s, we might add them before the conflicting {0,2} vectors are considered?
    #    No, the greedy process adds in descending priority.
    #    If we give high priority to (1,1,1), it gets added first.
    #    Then when (0,0,0) is considered, it might be rejected if it forms an AP with (1,1,1) and another already added vector?
    #    No, it's rejected if {v, u, w} is an AP where u, w are already in the set.
    #    If only (1,1,1) is in the set, (0,0,0) is valid.
    #    Then (2,2,2) is considered. (0,0,0), (1,1,1), (2,2,2) is an AP. So (2,2,2) is rejected.
    #    So we lose (2,2,2) but gain (1,1,1). Net change 0?
    #    
    #    However, (1,1,1) might conflict with other vectors in ways that allow adding *more* vectors overall?
    #    Or maybe we should prioritize vectors with an *even* number of 1s? Or specific patterns?
    #    
    #    The inspiration "Balanced digit distribution" scored 1.223.
    #    This implies that a heuristic based on balance (e.g., number of 1s) works better.
    #    Let's implement a score that increases with the number of 1s.
    #    score = number of 1s in the vector.
    #    This will put (1,1,...,1) at the top.
    #    
    #    Is there a better heuristic?
    #    Maybe score = sum of digits?
    #    For {0,1,2}^n, sum ranges from 0 to 2n.
    #    (1,1,...,1) has sum n.
    #    (0,0,...,0) has sum 0.
    #    (2,2,...,2) has sum 2n.
    #    If we prioritize middle sums, we get balanced vectors.
    #    
    #    Let's try: score = -abs(sum(vector) - n).
    #    This prioritizes vectors with sum close to n (i.e., balanced 0s, 1s, 2s).
    #    (1,1,...,1) has sum n, score 0 (highest).
    #    (0,0,...,0) has sum 0, score -n.
    #    (2,2,...,2) has sum 2n, score -n.
    #    (1,1,...,1,0) has sum n-1, score -1.
    #    
    #    This seems consistent with "balanced digit distribution".
    
    s = sum(vector)
    # Maximize closeness to n
    return -abs(s - n)
# EVOLVE-BLOCK-END
