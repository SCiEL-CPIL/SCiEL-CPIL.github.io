<<<<<<< SEARCH
def _generate_lattice_config():
    """Generate a hexagonal lattice initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    for a_deg in [30, 90, 150, 210]:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions


def _generate_random_config():
    """Generate a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions


def _generate_ring_config():
    """Generate a ring configuration with center and two rings."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    # Inner ring: 6 hexagons
    dist1 = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist1 * np.cos(a), dist1 * np.sin(a), 0.0])
    # Outer ring: 4 hexagons
    dist2 = 3.0 * np.sqrt(3)
    for a_deg in [30, 90, 150, 210]:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions
=======
def _generate_lattice_config():
    """Generate a hexagonal lattice initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    for a_deg in [30, 90, 150, 210]:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions


def _generate_random_config():
    """Generate a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions


def _generate_ring_config():
    """Generate a ring configuration with center and two rings."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist1 = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist1 * np.cos(a), dist1 * np.sin(a), 0.0])
    dist2 = 3.0 * np.sqrt(3)
    for a_deg in [30, 90, 150, 210]:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions


def _generate_compact_config():
    """Generate a compact configuration with varied angles."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    # 6 around center with slight angle variations
    dist1 = 1.8 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k + random.uniform(-10, 10))
        angle = random.uniform(0, 60)
        positions.append([dist1 * np.cos(a), dist1 * np.sin(a), angle])
    # 4 more in outer positions
    dist2 = 3.2 * np.sqrt(3)
    for a_deg in [30, 90, 150, 210]:
        a = np.radians(a_deg + random.uniform(-15, 15))
        angle = random.uniform(0, 60)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), angle])
    return positions
>>>>>>> REPLACE

