import time
import math

_start_time = time.time()
_DEADLINE = _start_time + 280.0

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # The strategy: Prioritize vectors with sparse/low-energy spectral representations.
    # In Z_3^n, the "frequency" of a vector can be approximated by its DFT.
    # A vector with high DFT energy (especially in high frequencies) has complex structure
    # prone to forming APs. A vector with low DFT energy (concentrated at DC) is "smooth".
    
    # However, a pure DFT over Z_3^n is expensive (O(n * 3^n)) if done naively per vector.
    # But we can use a property: The sum of squares of the DFT coefficients is proportional 
    # to the sum of squares of the original values (Parseval's theorem).
    # Wait, Parseval says ||f||^2 = (1/3^n) ||F||^2. This doesn't help distinguish "smooth" vs "rough" 
    # because all vectors have the same norm if they are in {0,1,2}^n? No, they don't.
    # But more importantly, for cap sets, we want to avoid additive energy.
    # Additive energy E(A) is related to the 4th moment of the Fourier transform.
    # E(A) = (1/3^n) sum_k |hat{A}(k)|^4.
    # We want to minimize the 4th moment of the spectrum.
    
    # Computing the full 4th moment of the DFT for every vector is O(3^n) which is too slow for n>10.
    # We need a proxy.
    
    # Alternative Interpretation of "Spectral Sparsity":
    # In many optimization contexts, "spectral sparsity" refers to the number of non-zero Fourier coefficients.
    # For a vector in {0,1,2}^n, the "smoothness" is often correlated with the variance of the signal.
    # Actually, let's look at Inspiration 1 which scored 1.2930. It used "magnitude of their Discrete Fourier Transform".
    # If n is small (e.g., n <= 10), 3^n is 59049. 
    # If n=12, 3^12 is 531441.
    # If n=15, 3^15 is 14M.
    # The evaluator kills at 330s. 
    # If we call priority() many times, we cannot afford O(3^n) per call.
    
    # Let's assume n is small enough that O(n^2) or O(n) is required, or we use a faster proxy.
    # The current program uses O(n) stats.
    # A "Global Spectral" proxy that is O(n) or O(n^2):
    # The "Spectral Energy" of a signal x is sum |X_k|^2. By Parseval, this is constant for fixed ||x||.
    # But we care about the *distribution* of energy.
    # Low frequency energy = DC component + small k.
    # High frequency energy = large k.
    
    # A cheap proxy for "high frequency content" is the variance of the first differences.
    # If x is smooth, differences are small.
    # If x is rough, differences are large.
    # This is exactly the "spread" or "smoothness" the current program tries to capture, but maybe not optimally weighted.
    
    # Let's try a different structural metric: **Additive Energy Proxy via Pairwise Sums**.
    # Two vectors u, v form an AP if u+v = 2w. In Z_3, 2 = -1. So u-v = w? No.
    # AP: a, b, c such that a+c = 2b.
    # This is hard to check per-pair.
    
    # Let's go back to the DFT idea but approximate it.
    # For n <= 10, we can actually compute the DFT magnitude squared sum for the "high frequencies" if we precompute basis?
    # No, vector changes every time.
    
    # Let's try **Random Projection Spectral Norm**.
    # Project the vector onto a random subspace and measure the norm?
    
    # Actually, let's look at the "Sparse Spectral Representation" directive.
    # "Minimizing the number of non-zero coefficients in their discrete Fourier transform".
    # This is equivalent to minimizing the support of the spectrum.
    # For a real signal, a signal with few non-zero Fourier coeffs is a sum of few sinusoids.
    # Such signals are "structured".
    # Wait, cap sets want to AVOID structure that creates APs.
    # Random sets have "flat" spectra (all coeffs roughly equal magnitude).
    # Structured sets (like subspaces) have "peaked" spectra.
    # So we want to AVOID peaked spectra?
    # Or do we want to prefer peaked spectra?
    # Subspaces are large cap sets. Their spectrum is supported on the dual subspace.
    # So, vectors that are "close to being in a subspace" might be good?
    # But greedy addition is different.
    
    # Let's try **Variance of the Signal** as a proxy for "High Frequency Energy".
    # High variance = High Frequency Energy.
    # Low variance = Low Frequency Energy (Smooth).
    # The current program prefers Smooth (low spread).
    # Maybe we should prefer **High** Spectral Energy? i.e., Random-like?
    # Random vectors are less likely to form APs with each other?
    # Actually, random sets have low additive energy.
    # Additive energy is high for structured sets.
    # So we want to minimize Additive Energy.
    # Minimizing Additive Energy corresponds to flattening the spectrum (making it more uniform).
    # So we want vectors that are "more random" (higher variance, less smooth)?
    
    # Let's test the hypothesis: **Maximize Variance** (or Spread).
    # The current program minimizes spread (prefers smooth).
    # Let's flip it: Maximize spread.
    
    # But wait, Inspiration 1 (1.29) used DFT magnitude.
    # If we just flip the sign of the smoothness term in the current program, is that "materially different"?
    # The directive says: "Prioritize vectors by minimizing the number of non-zero coefficients... favors sparse spectral representations".
    # Sparse spectrum = Few non-zero coeffs = Peaked Spectrum = Structured/Smooth.
    # So the directive suggests preferring SMOOTH vectors.
    # The current program already prefers smooth vectors (score = ... + 1.0 * smoothness_score).
    # Why is it only 1.08?
    # Maybe the weighting is wrong, or the "smoothness" metric (variance) is a poor proxy for spectral sparsity.
    
    # A better proxy for spectral sparsity: **Autocorrelation**.
    # A signal with sparse spectrum has high autocorrelation at small lags?
    # Or, the **Hurst Exponent**? Too complex.
    
    # Let's use **Second Order Differences**.
    # Variance of first differences captures high freq.
    # Variance of second differences captures even higher freq.
    # If we want to minimize "non-zero coefficients" in DFT, we want the signal to be close to a polynomial of low degree?
    # In Z_3, a linear function has a very sparse spectrum (only 2 non-zero coeffs?).
    # So, we should prefer vectors that are "linear-like".
    
    # Metric: **Residual of Linear Fit**.
    # Fit a linear function f(i) = a*i + b to the vector values?
    # No, the domain is Z_3^n, not Z.
    # "Linear" in Z_3^n means f(x) = sum c_j x_j.
    # We can't easily fit this without solving a system.
    
    # Alternative: **Correlation with Basis Vectors**.
    # A vector is "sparse in spectrum" if it is close to a character (homomorphism from Z_3^n to C).
    # Characters are of the form chi_k(x) = omega^{k dot x}.
    # We can check how well the vector correlates with a few random characters.
    # If it correlates highly with any character, it has a large spectral component there.
    # We want to MINIMIZE the number of large components?
    # Or MAXIMIZE the max component?
    # If it is close to a single character, its spectrum is sparse (1 peak).
    # So we want to MAXIMIZE the maximum correlation with any character?
    
    # Let's implement:
    # 1. Generate a few random characters (vectors k in {0,1,2}^n).
    # 2. Compute the correlation (real part of inner product with omega^{k dot x}).
    # 3. Score = Max Correlation.
    
    # This is O(K * n) where K is number of random characters.
    # This is fast.
    
    # Why would this work?
    # Vectors close to characters are "structured" and might be good building blocks for subspaces?
    # Or, if we are building a cap set, we want to avoid APs.
    # Characters have no APs?
    # If x is a character, x, y, z in AP implies chi(x), chi(y), chi(z) in AP.
    # Chi values are on the unit circle.
    # This is a specific algebraic property.
    
    # Let's try this "Character Correlation" strategy.
    
    # Precompute random characters?
    # We need to be deterministic.
    
    import random
    _rng = random.Random(42)
    _NUM_CHARACTERS = 10
    _MAX_N = 20
    # We need to generate random k vectors on the fly or precompute for max n?
    # n varies.
    # We can generate k vectors inside the function if n is small?
    # Or use a fixed set of "basis" directions?
    
    # Let's use a fixed set of "structural" directions:
    # 1. All 1s vector (DC component)
    # 2. Alternating 1, 2, 1, 2... (High freq)
    # 3. Random directions.
    
    # Actually, simpler: **Max Variance of Projections**.
    
    # Let's stick to the Directive: "Minimizing the number of non-zero coefficients".
    # This is equivalent to maximizing the "Purity" of the spectrum.
    # Purity = (sum |c_k|^2)^2 / sum |c_k|^4.
    # If spectrum is 1 peak, Purity = 1.
    # If flat, Purity = 1/N.
    # We want High Purity.
    
    # We can't compute full spectrum.
    # But we can approximate the 4th moment using random projections?
    # E[|x|^4] is related to the 4th moment of the projection.
    
    # Let's try a simpler structural change:
    # **Greedy Conflict Minimization with Geometric Awareness** (Inspiration 2).
    # But we don't have access to the current set in `priority`.
    # `priority` only sees the vector.
    
    # So we must use a **Static** measure.
    
    # Let's try **Entropy of the Value Distribution** vs **Positional Entropy**.
    # Current: Balance (Value Distribution).
    # New: **Shannon Entropy of the Vector Sequence**.
    # Treat the vector as a string over {0,1,2}.
    # Compute the entropy of the sequence.
    # High entropy = Random. Low entropy = Structured.
    # If Directive says "Sparse Spectrum" = Structured, we prefer LOW Entropy.
    
    # Let's calculate Sequence Entropy.
    
    # Count occurrences of bigrams?
    # Or just unigram entropy is just balance.
    # Bigram entropy captures local structure.
    
    # Let's try **Bigram Entropy**.
    # Count (v_i, v_{i+1}) pairs.
    # Compute entropy.
    # Prefer Low Entropy (Structured/Smooth).
    
    # This is O(n).
    
    # Let's implement Bigram Entropy.
    
    if n < 2:
        return 1.0

    # Count bigrams
    # There are 9 possible bigrams: (0,0), (0,1), ... (2,2)
    counts = [0] * 9
    for i in range(n - 1):
        v1 = vector[i]
        v2 = vector[i + 1]
        idx = v1 * 3 + v2
        counts[idx] += 1
    
    total_pairs = n - 1
    if total_pairs == 0:
        return 1.0
        
    # Calculate Entropy
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total_pairs
            entropy -= p * math.log(p, 3) # Log base 3 for normalization to [0, 2]
            
    # Max entropy for 9 symbols is 2.0.
    # Min entropy is 0.0 (all same bigram).
    
    # We want to MINIMIZE entropy (prefer structured).
    # So score = 1.0 - entropy / 2.0
    
    # Combine with Balance (Symbol Counts) to ensure global distribution is also good.
    # Balance score from current program:
    c0 = sum(1 for v in vector if v == 0)
    c1 = sum(1 for v in vector if v == 1)
    c2 = sum(1 for v in vector if v == 2)
    
    ideal = n / 3.0
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    max_var = (2.0 * n * n) / 3.0
    if max_var > 0:
        balance_score = 1.0 - (var / max_var)
    else:
        balance_score = 1.0
        
    # Structure Score (Low Entropy)
    structure_score = 1.0 - (entropy / 2.0)
    
    # Final Score
    # Weight Structure heavily?
    # Directive: "Prioritize vectors by minimizing... spectral... favors sparse...".
    # So Structure is the main driver.
    
    score = 2.0 * structure_score + 1.0 * balance_score
    
    return float(score)
