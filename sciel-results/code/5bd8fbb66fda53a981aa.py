import numpy as np
import time

def quote(t, inventory, mid, params):
    """
    Analytical market-making policy from continuous-time HJB first-order condition.
    
    The optimal depth balances expected fill profit against inventory risk.
    For a market maker with inventory q, the HJB value function V(t,q) satisfies:
      -dV/dt = max_d { A*exp(-k*d)*d + A*exp(-k*d)*d - gamma/2*sigma^2*q^2*dt 
                       + sigma^2/2 * d2V/dq2 * dt }
    
    Taking the FOC with respect to depth d for each side:
    The optimal bid depth (when q < Q) and ask depth (when q > -Q) satisfy:
      k * d * exp(-k*d) + 1 = lambda(t,q)
    where lambda depends on the marginal value of inventory.
    
    For the symmetric case with quadratic inventory cost, the optimal depth is:
      d*(q,t) = W(1 + gamma*sigma^2*q*dt_remaining_factor) / k
    where W is the Lambert W function.
    
    More precisely, for the discrete model, the optimal depth that maximizes
    expected immediate reward minus inventory risk adjustment is:
      d* = argmax_d { A*exp(-k*d)*d - (gamma/2)*sigma^2*(q+1)^2*dt + (gamma/2)*sigma^2*q^2*dt }
           = argmax_d { A*exp(-k*d)*d - gamma*sigma^2*q*dt - gamma/2*sigma^2*dt }
    
    The FOC: -A*k*exp(-k*d)*(d) + A*exp(-k*d) = 0  =>  d = 1/k (risk-neutral)
    But with inventory risk, the effective cost shifts the optimum.
    
    A cleaner derivation: the value of being at inventory q after a bid fill is
    V(t+dt, q+1) vs V(t+dt, q). The marginal cost of holding more inventory is
    approximately gamma*sigma^2*q*dt. This acts as an additional cost per unit depth.
    
    For the bid side (increases inventory):
      max_d A*exp(-k*d)*d - gamma*sigma^2*q*dt  (for q >= 0)
      FOC: A*exp(-k*d)*(1 - k*d) = 0 => d = 1/k if q >= 0
      But the inventory penalty shifts: effectively we're maximizing
      A*exp(-k*d)*d - c*q where c = gamma*sigma^2*dt
      This doesn't change the FOC for d since c*q is constant in d.
      
    The key insight: in the discrete model, the optimal depth is always 1/k 
    (the risk-neutral point) UNLESS inventory is at bounds. The inventory risk 
    affects WHETHER to quote, not the depth itself, in this specific model structure.
    
    However, the inventory risk DOES affect the quote decision: when |q| is large,
    the expected cost of taking on more inventory exceeds the expected fill profit.
    
    For bid (q -> q+1): expected profit = A*exp(-k*d)*d*dt
    Expected inventory cost increase = gamma/2*sigma^2*((q+1)^2 - q^2)*dt = gamma*sigma^2*(q+0.5)*dt
    Quote bid if: A*exp(-k*d)*d*dt > gamma*sigma^2*(q+0.5)*dt
    At d=1/k: A/k > gamma*sigma^2*(q+0.5)
    
    For ask (q -> q-1): expected profit = A*exp(-k*d)*d*dt
    Expected inventory cost decrease = gamma/2*sigma^2*(q^2 - (q-1)^2)*dt = gamma*sigma^2*(q-0.5)*dt
    Quote ask if: A*exp(-k*d)*d*dt > gamma*sigma^2*(q-0.5)*dt  (for q < 0, this is negative cost = benefit)
    
    Actually, let me reconsider. The objective is:
      sum of depths earned - sum over steps of (gamma/2)*sigma^2*q^2*dt
    
    So the inventory penalty is (gamma/2)*sigma^2*q^2*dt per step, charged on q BEFORE fills.
    
    If I quote bid at depth d: with prob p_bid = 1-exp(-A*exp(-k*d)*dt), I get depth d 
    and inventory goes from q to q+1.
    If I quote ask at depth d: with prob p_ask = 1-exp(-A*exp(-k*d)*dt), I get depth d
    and inventory goes from q to q-1.
    
    The expected immediate reward from quoting both sides at depth d:
      d * (1 - exp(-A*exp(-k*d)*dt)) + d * (1 - exp(-A*exp(-k*d)*dt))
      = 2*d*(1 - exp(-A*exp(-k*d)*dt))
    
    The inventory penalty this step is (gamma/2)*sigma^2*q^2*dt regardless of fills.
    
    But the FUTURE inventory state matters. After this step, inventory is:
      q+1 with prob p_bid*(1-p_ask)
      q-1 with prob p_ask*(1-p_bid)  
      q with prob (1-p_bid)*(1-p_ask) + p_bid*p_ask (both hit or neither)
      q+2 with prob p_bid*p_ask (if both can hit, but they're independent)
      q-2 with prob p_bid*p_ask
    
    Wait, bid and ask fills are independent. So:
      P(q+1) = p_bid*(1-p_ask)
      P(q-1) = (1-p_bid)*p_ask
      P(q) = (1-p_bid)*(1-p_ask) + p_bid*p_ask
      P(q+2) = p_bid*p_ask  -- NO, bid gives q+1, ask gives q-1, so both give q
      Actually: if bid fills, q->q+1. If ask fills, q->q-1. Both independent.
      So: q+1 with p_bid*(1-p_ask), q-1 with (1-p_bid)*p_ask, q with (1-p_bid)*(1-p_ask)+p_bid*p_ask, 
      q+2 with p_bid*p_ask, q-2 with p_bid*p_ask? No:
      bid fill: q->q+1, ask fill: q->q-1. Both: q->q+1-1=q. Neither: q.
      So: q+1: p_bid*(1-p_ask), q-1: (1-p_bid)*p_ask, q: (1-p_bid)*(1-p_ask)+p_bid*p_ask
    
    The future value depends on the resulting inventory. For the optimal policy,
    we need V(t+dt, q') for each possible q'.
    
    For a first-order approximation, the marginal value of inventory is:
      dV/dq ≈ gamma*sigma^2*q (from the quadratic term)
    
    So the cost of increasing inventory by 1 is approximately gamma*sigma^2*q*dt (times dt for the step).
    
    The optimal bid depth maximizes:
      d*(1-exp(-A*exp(-k*d)*dt)) - gamma*sigma^2*q*dt*(1-exp(-A*exp(-k*d)*dt))
      = (1-exp(-A*exp(-k*d)*dt)) * (d - gamma*sigma^2*q*dt)
    
    Let p = 1-exp(-A*exp(-k*d)*dt). For small dt, p ≈ A*exp(-k*d)*dt.
    
    Maximize: A*exp(-k*d)*dt * (d - gamma*sigma^2*q*dt)
    FOC w.r.t. d: A*dt * [-k*exp(-k*d)*(d - c) + exp(-k*d)*1] = 0 where c = gamma*sigma^2*q*dt
    => -k*(d-c) + 1 = 0
    => d = c + 1/k = gamma*sigma^2*q*dt + 1/k
    
    For ask: the inventory change is q->q-1, so the marginal cost is -gamma*sigma^2*q*dt (benefit if q>0)
    Maximize: A*exp(-k*d)*dt * (d + gamma*sigma^2*q*dt)  [since reducing inventory when q>0 is beneficial]
    FOC: -k*(d+c) + 1 = 0 => d = 1/k - c = 1/k - gamma*sigma^2*q*dt
    
    So:
      bid_depth = 1/k + gamma*sigma^2*q*dt  (for q >= 0, this increases depth when long)
      ask_depth = 1/k - gamma*sigma^2*q*dt  (for q > 0, this decreases depth when long)
    
    Wait, let me redo this more carefully.
    
    For bid (q -> q+1): the inventory penalty NEXT step will be (gamma/2)*sigma^2*(q+1)^2*dt vs current (gamma/2)*sigma^2*q^2*dt.
    Difference: (gamma/2)*sigma^2*((q+1)^2 - q^2)*dt = (gamma/2)*sigma^2*(2q+1)*dt = gamma*sigma^2*(q+0.5)*dt
    
    So the effective cost of a bid fill is gamma*sigma^2*(q+0.5)*dt in addition to the depth earned.
    
    Expected net from bid at depth d:
      p_bid * (d - gamma*sigma^2*(q+0.5)*dt)
    where p_bid = 1 - exp(-A*exp(-k*d)*dt)
    
    For small dt: ≈ A*exp(-k*d)*dt * (d - gamma*sigma^2*(q+0.5)*dt)
    
    FOC: d/dd [exp(-k*d)*(d - c)] = 0 where c = gamma*sigma^2*(q+0.5)*dt
    exp(-k*d)*[1 - k*(d-c)] = 0
    => d = c + 1/k
    
    For ask (q -> q-1): inventory penalty change = (gamma/2)*sigma^2*((q-1)^2 - q^2)*dt = (gamma/2)*sigma^2*(-2q+1)*dt = -gamma*sigma^2*(q-0.5)*dt
    So the benefit of an ask fill is -gamma*sigma^2*(q-0.5)*dt (positive when q > 0.5)
    
    Expected net from ask at depth d:
      p_ask * (d + gamma*sigma^2*(q-0.5)*dt)  [the + is because reducing inventory reduces penalty]
    
    FOC: d = -gamma*sigma^2*(q-0.5)*dt + 1/k
    
    So:
      bid_depth(q) = 1/k + gamma*sigma^2*(q+0.5)*dt
      ask_depth(q) = 1/k - gamma*sigma^2*(q-0.5)*dt
    
    We should clip to [0, inf) and set to inf if the depth would be negative (don't quote).
    Also, at q=Q, no bid; at q=-Q, no ask.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_paths = len(inventory)
    
    # Base depth
    base = 1.0 / k
    
    # Inventory adjustment
    inv = inventory.astype(np.float64)
    
    # Bid depth: 1/k + gamma*sigma^2*(q+0.5)*dt
    bid_depth = base + gamma * sigma**2 * (inv + 0.5) * dt
    
    # Ask depth: 1/k - gamma*sigma^2*(q-0.5)*dt
    ask_depth = base - gamma * sigma**2 * (inv - 0.5) * dt
    
    # Clip to non-negative; negative means don't quote (inf)
    bid_depth = np.where(bid_depth > 0, bid_depth, np.inf)
    ask_depth = np.where(ask_depth > 0, ask_depth, np.inf)
    
    # At Q, no bid; at -Q, no ask
    bid_depth = np.where(inv >= Q, np.inf, bid_depth)
    ask_depth = np.where(inv <= -Q, np.inf, ask_depth)
    
    # Ensure minimum valid depth > -3/k (should always be satisfied since we clip to >= 0)
    # Values below -3/k are invalid, but our depths are >= 0 or inf, so we're fine.
    
    return bid_depth, ask_depth
