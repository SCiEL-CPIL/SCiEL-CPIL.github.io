import math
import time
import numpy as np
from scipy.optimize import minimize

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    # Same as before
    pass

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _synthesize_known(name, qubits, budget):
    if name == "CNOT":
        return [["cx", 0, 1]]
    if name == "SWAP":
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    return None

def _synthesize_generic(problem, time_limit):
    # ... optimization logic ...
    pass

def synthesize(problem):
    name = problem.get("name", "generic")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Try known
    known_gates = _synthesize_known(name, qubits, budget)
    if known_gates is not None:
        return known_gates
        
    # Otherwise generic
    target = _target(problem)
    # Set time limit per gate, e.g., 40s
    return _synthesize_generic(problem, target, 40.0)
