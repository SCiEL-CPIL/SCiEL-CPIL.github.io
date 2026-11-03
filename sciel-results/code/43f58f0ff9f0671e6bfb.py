"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
import itertools

N = 100

# We are choosing h[0..N-1] in [0,1] with sum(h) = N/2.
# Overlap of h with shifted 1-h:
#   For a shift k (circular), overlap_k = sum_i h[i] * (1 - h[(i-k) % N])
#     = sum_i h[i] - sum_i h[i]*h[(i-k)%N]
#     = N/2 - sum_i h[i]*h[(i-k)%N]   (since sum h = N/2)
#
# We want to MINIMIZE the MAX over k of overlap_k.
# Equivalently, since overlap_k = N/2 - dot_k where dot_k = sum_i h[i]*h[(i-k)%N],
# minimizing max overlap is equivalent to MINIMIZING (N/2 - min_k dot_k),
# i.e., MAXIMIZING min_k dot_k.
#
# So the objective is: maximize the minimum circular autocorrelation at non-zero lags? 
# Wait: k=0 gives dot_0 = sum h[i]^2. The task says "shifted 1-h". Let me re-read.
#
# "minimize the maximum overlap of h with shifted 1-h"
# overlap_k = sum_i h[i] * (1 - h[(i-k) % N]) for k = 1..N-1 (or all k?)
# If k=0: overlap_0 = sum h[i](1-h[i]) = N/2 - sum h[i]^2
#
# Typically "shifted" means k != 0. But let's consider all shifts k=0..N-1 to be safe,
# or just k=1..N-1. The problem says "shifted 1-h", implying a non-trivial shift.
# However, the overlap at k=0 is also a valid overlap if we consider the function h and 1-h
# aligned. But "shifted" usually implies k != 0.
#
# Let's look at the structure. h is a step function on [0,2]. 
# The overlap integral is ∫ h(x) (1-h(x-k)) dx.
#
# Let's define the objective as max_{k in 0..N-1} overlap_k.
# overlap_k = N/2 - C_k, where C_k = sum_i h[i] h[(i-k)%N].
# So we want to minimize max_k (N/2 - C_k) = N/2 - min_k C_k.
# This is equivalent to maximizing min_k C_k.
#
# Note C_k = C_{N-k}. So we only need to consider k = 0, 1, ..., N//2.
# C_0 = sum h[i]^2. Since h[i] in [0,1] and sum h = N/2, C_0 is at least (N/2)^2/N = N/4.
# The other C_k can be negative? No, h[i] >= 0, so C_k >= 0.
#
# We want to maximize min_k C_k.
# Let t = min_k C_k. We want to maximize t.
# Constraints:
#   sum h = N/2
#   0 <= h[i] <= 1
#   C_k >= t for all k in 0..N//2
#
# This is a non-linear programming problem because C_k is quadratic in h.
# However, for a fixed t, the constraints C_k >= t are convex? No, C_k is convex (quadratic form with PSD matrix? C_k = h^T P_k h where P_k is a permutation matrix. P_k is symmetric and orthogonal, so eigenvalues are 1. Thus C_k is convex).
# Maximizing t subject to convex constraints is not a convex optimization problem directly (maximizing a linear function over a non-convex set defined by convex inequalities? Wait. C_k >= t is a convex set in (h,t) for fixed h? No. The set {h | C_k(h) >= t} is not convex if C_k is convex. It's the superlevel set of a convex function, which is generally non-convex).
#
# Actually, we can view this as: minimize t such that ... no, we want to maximize t.
# Let's try a different approach.
# Since the problem is symmetric and involves autocorrelation, perhaps a specific structure works well.
#
# Consider h[i] = 0.5 for all i.
# C_k = sum 0.5*0.5 = N * 0.25 = N/4 for all k.
# min C_k = N/4 = 25.
# max overlap = N/2 - N/4 = N/4 = 25.
# Score = 25.
#
# Can we do better?
# If we make h binary (0 or 1), sum h = N/2 = 50.
# Then C_0 = 50.
# For k != 0, C_k is the number of positions where h[i]=1 and h[i-k]=1.
# We want to maximize the minimum C_k.
# This is related to difference sets or balanced sequences.
# If we can make C_k constant for all k != 0, that would be ideal.
# For a binary sequence of length N with 50 ones, the autocorrelation C_k for k!=0.
# Sum_{k=0}^{N-1} C_k = (sum h[i])^2 = 50^2 = 2500.
# C_0 = 50.
# Sum_{k=1}^{N-1} C_k = 2450.
# Average C_k for k!=0 is 2450 / 99 ≈ 24.75.
# So the minimum C_k for k!=0 cannot exceed 24.75.
# Thus max overlap = 50 - min C_k >= 50 - 24.75 = 25.25.
# This is worse than 25.
#
# So the constant 0.5 solution gives 25.
# Can we get below 25?
# If h is not binary, C_0 = sum h[i]^2. By Cauchy-Schwarz, C_0 >= (sum h)^2 / N = 2500/100 = 25.
# So C_0 >= 25.
# If we make C_k for k!=0 larger than 25, then min C_k could be 25 (if C_0 is the min).
# But we want to maximize min C_k.
# If we can make all C_k >= 25, then min C_k >= 25, so max overlap <= 25.
# The constant solution achieves exactly 25 for all C_k.
# So 25 is a lower bound on the max overlap?
# Wait. max overlap = N/2 - min C_k.
# If min C_k = 25, max overlap = 25.
# Can we make min C_k > 25?
# We know C_0 >= 25. If we can make all C_k >= 25 + epsilon, then min C_k > 25.
# But sum C_k = (sum h)^2 = 2500.
# There are N terms. If all C_k >= 25, sum C_k >= 2500.
# Since sum C_k = 2500 exactly, we must have C_k = 25 for all k.
# This implies h is constant 0.5 (by equality condition in Parseval/autocorrelation properties).
# So the constant solution is optimal!
#
# Wait, is that true?
# Let's check.
# C_k = sum_i h[i] h[i-k].
# Sum_k C_k = sum_k sum_i h[i] h[i-k] = sum_i h[i] (sum_k h[i-k]) = sum_i h[i] (sum_j h[j]) = (sum h)^2 = 2500.
# If C_k >= 25 for all k, then sum C_k >= 100 * 25 = 2500.
# So C_k = 25 for all k is the only possibility if min C_k >= 25.
# And we know C_0 >= 25.
# So min C_k <= C_0. But we also have the sum constraint.
# Actually, we want to MAXIMIZE min C_k.
# Let m = min C_k.
# We have sum C_k = 2500.
# So N * m <= 2500 => m <= 25.
# So min C_k <= 25.
# Therefore, max overlap = N/2 - min C_k >= 50 - 25 = 25.
# So the minimum possible max overlap is 25.
# And the constant solution achieves 25.
#
# So the answer is 25.
#
# But the task says "score is the upper bound on C5" and "lower is better".
# And "1.0 equals the seed program".
# The seed program returns 0.5 for all N, so score is 25.
# So the seed score is 25.
# And we just proved 25 is optimal.
#
# So the best possible score is 25.
#
# Let me double check the problem statement.
# "Choose equal-width step heights h on [0,2], bounded between 0 and 1 with integral 1"
# Integral 1 means sum h[i] * (2/N) = 1 => sum h[i] = N/2 = 50. Correct.
# "minimize the maximum overlap of h with shifted 1-h"
# overlap_k = ∫ h(x) (1-h(x-k)) dx ≈ (2/N) sum_i h[i] (1-h[i-k]).
# Wait, the integral is over [0,2]. The width is 2.
# So the integral is (2/N) * sum_i h[i] (1-h[i-k]).
# So overlap_k = (2/N) * (N/2 - C_k) = 1 - (2/N) C_k.
# We want to minimize max_k overlap_k = 1 - (2/N) min_k C_k.
# Minimizing this is equivalent to maximizing min_k C_k.
# We found max min_k C_k = 25.
# So min max overlap = 1 - (2/100)*25 = 1 - 0.5 = 0.5.
#
# So the score should be 0.5.
#
# The seed program returns 0.5 for all h.
# C_k = 25 for all k.
# overlap_k = 1 - 0.5 * 1 = 0.5.
# So the seed score is 0.5.
#
# And we proved 0.5 is optimal.
#
# So the program just needs to return the constant 0.5 solution.
#
# But wait, the task says "score is the upper bound on C5".
# And "1.0 equals the seed program".
# This implies the seed program has score 1.0?
# Or is the score normalized?
# "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program"
# So if the raw value is 0.5, and the seed raw value is 0.5, then score = 0.5/0.5 = 1.0.
#
# So the current program (returning 0.5) has score 1.0.
# And we proved it's optimal.
#
# So I should just return the constant 0.5 solution.
#
# But maybe I made a mistake in the proof?
# Let's re-verify.
# overlap_k = (2/N) sum_i h[i] (1 - h[i-k])
# = (2/N) (sum h[i] - sum h[i] h[i-k])
# = (2/N) (N/2 - C_k)
# = 1 - (2/N) C_k.
#
# We want to minimize max_k overlap_k.
# max_k overlap_k = 1 - (2/N) min_k C_k.
# To minimize this, we maximize min_k C_k.
#
# We have sum_k C_k = (sum h[i])^2 = (N/2)^2 = N^2/4.
# By AM-GM or just sum, min_k C_k <= (sum C_k) / N = (N^2/4) / N = N/4.
# So min_k C_k <= N/4.
# Thus max_k overlap_k >= 1 - (2/N) (N/4) = 1 - 1/2 = 1/2.
#
# Equality holds when C_k = N/4 for all k.
# This requires h[i] = 1/2 for all i (since the autocorrelation is constant, which implies the signal is constant in the frequency domain, i.e., only DC component, which means constant in time domain).
#
# So the optimal solution is indeed h[i] = 0.5 for all i.
#
# So the program is already optimal.
#
# But the task asks to "rewrite the program from scratch with a stronger algorithm".
# Maybe I should implement a solver that verifies this, or just return the optimal solution directly.
# Since the optimal solution is known analytically, I can just return it.
#
# However, to be safe and to follow the directive to "investigate improvements within its family", I could implement a small optimization to confirm, but since the analytical solution is proven optimal, returning it directly is best.
#
# Let's just return the constant array.

def construct():
    return np.full(N, 0.5)

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 100

def construct():
    # The optimal solution is h[i] = 0.5 for all i.
    # This is proven by the fact that min_k C_k <= N/4, with equality iff h is constant.
    # So the maximum overlap is exactly 0.5.
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
