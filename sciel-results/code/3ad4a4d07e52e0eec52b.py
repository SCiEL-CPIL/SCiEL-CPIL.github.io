"""Rewritten program: use Probability of Improvement (POI) with exploration bonus,
combined with a small random restart / diversity term to escape local optima.
Also consider Upper Confidence Bound (UCB) as an alternative acquisition.
Since the task says 'higher meaning evaluate next', we want to maximize.

Let's design a hybrid: primarily UCB with a tunable beta, but also add a small
random exploration component to ensure diversity. However, the signature is fixed:
acquisition(mu, sigma, f_best) -> 2000 scores.

Key insight: The current program uses EI. Let's try UCB, which often performs better
in practice for multimodal functions like Michalewicz. UCB = mu + beta * sigma.
But wait: we are minimizing. So for minimization, we want low mu and high uncertainty.
UCB for minimization is: -mu + beta * sigma. Or equivalently, we can maximize -mu + beta*sigma.

Let's also consider that Michalewicz 2D is multimodal. A pure UCB might get stuck.
Adding a small amount of random exploration (e.g., adding a small random noise to the score)
can help, but the function is deterministic given mu, sigma, f_best. We can't use randomness
inside acquisition because it would make the score non-deterministic and potentially invalid
if the evaluator expects consistent behavior? Actually, the contract says "Return 2000 finite scores".
It doesn't say they must be deterministic. But typically, acquisition functions are deterministic
given their inputs. Using randomness could lead to different paths each time, which might be good
for exploration but could also be seen as unstable. Let's stick to deterministic.

Alternative: Use a combination of UCB and EI, or use a different acquisition function like
Probability of Improvement (POI): P(f(x) < f_best) = Phi((f_best - mu) / sigma).
POI is more exploitative than EI. UCB is more exploratory.

Let's try UCB with a beta that adapts or is fixed. A common choice for beta is sqrt(2 * log(n))
where n is the number of evaluations. But we don't have n directly. We can estimate it or use a fixed beta.
Since we have 30 evaluations total, and we start with 5, we have 25 acquisition steps.
A fixed beta of around 2-5 might work. Let's try beta = 3.0.

But to be more robust, let's consider a weighted sum of UCB and a small random-like term derived from the indices?
No, that's hacky. Let's just use UCB.

Wait, the directive says "design a new program ... from an approach family that none of the current programs use".
The current program uses EI. So UCB is a different family.

Let's implement UCB for minimization:
score = -mu + beta * sigma

We need to handle sigma = 0. If sigma is very small, we should avoid division issues, but here we're multiplying, so it's fine.
However, if sigma is 0, the term beta*sigma is 0, and we just pick the lowest mu. That's fine.

Let's also consider that the GP is standardized, so mu and sigma are in objective units.
The minimum value is -1.8013. f_best will be around that value or higher (worse).

Let's try beta = 2.5. This is a moderate exploration constant.

To make it even stronger, we could use a variant of UCB that accounts for the fact that we are in a bounded domain.
But the coordinates are not passed, so we can't do domain-specific adjustments.

Another idea: Use a combination of UCB and POI, but weighted.
score = w * (-mu + beta * sigma) + (1-w) * Phi((f_best - mu) / sigma)
But POI is between 0 and 1, while UCB can be negative or positive. They are on different scales.
We should normalize or use one or the other.

Let's stick with pure UCB for now, as it's a well-known alternative to EI.

Implementation:
import numpy as np
from scipy.stats import norm  # not needed for UCB, but keep if we want to mix.

def acquisition(mu, sigma, f_best):
    beta = 2.5
    # For minimization, we want to maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores

But wait, what if sigma is very large? UCB might over-explore. What if mu is very negative? 
UCB will prefer those. That's good.

Let's test this mentally:
- If mu is low (good) and sigma is low, score is high (good).
- If mu is high (bad) but sigma is very high, score might be high, encouraging exploration.
- If mu is moderate and sigma is moderate, score is moderate.

This seems reasonable.

However, to be even more robust, we could use a beta that decreases over time. But we don't have a time index.
We could use the number of unique values in mu or something, but that's not reliable.

Let's try a fixed beta = 2.0, which is a common default.

Actually, let's consider that the GP is Matern-5/2 with length scale 0.2. This means the GP is quite smooth.
UCB with a fixed beta should work well.

Let's go with beta = 2.0.

One more thought: The task says "higher meaning evaluate next". So we return scores, and the caller picks the max.
Our scores are -mu + beta*sigma. Higher is better. Correct.

Let's write the code.
"""
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # We want to maximize: -mu + beta * sigma
    beta = 2.0
    scores = -mu + beta * sigma
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
